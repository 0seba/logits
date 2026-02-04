"""
Improved Logit Extraction Pipeline

Extracts nucleus (top-p) and sampled logits from a language model for knowledge 
distillation, with streaming upload to HuggingFace Hub and automatic resume support.

The nucleus extraction uses top-p sampling: for each token, we extract the minimum
set of top logits whose cumulative probability exceeds p (default 0.98), with a hard
maximum of 100 elements. This results in variable-length extraction that saves storage
when the distribution is concentrated.

Usage:
    python extract_logits.py --model Qwen/Qwen3-4B-Instruct-2507 \
                             --dataset MegaScience/MegaScience \
                             --output seba/MegaScience-Qwen3-4B-Logits \
                             --top-p 0.98 --top-p-max 100 --sampled-n 24

    # Resume from interruption:
    python extract_logits.py --model Qwen/Qwen3-4B-Instruct-2507 \
                             --dataset MegaScience/MegaScience \
                             --output seba/MegaScience-Qwen3-4B-Logits \
                             --resume

    # Process full dataset without sampling:
    python extract_logits.py --model Qwen/Qwen3-4B-Instruct-2507 \
                             --dataset MegaScience/MegaScience \
                             --output seba/MegaScience-Qwen3-4B-Logits \
                             --full-dataset
"""

from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import argparse
import glob
import io
import json
import os
import signal
import sys
import threading
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from numpy.random import PCG64
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Timing Metrics
# =============================================================================


@dataclass
class BatchTimingMetrics:
    """Timing metrics for a single batch."""
    num_tokens: int = 0
    model_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0

    @property
    def model_tokens_per_sec(self) -> float:
        if self.model_time_ms > 0:
            return self.num_tokens / (self.model_time_ms / 1000.0)
        return 0.0

    @property
    def total_time_ms(self) -> float:
        return self.model_time_ms + self.postprocess_time_ms


# =============================================================================
# Helper: Index Packing
# =============================================================================


def pack_indices(indices: np.ndarray) -> tuple[np.ndarray, bytes]:
    """
    Pack an array of indices (int32) into:
    - low_bits: uint16 array (indices & 0xFFFF)
    - high_bits: bytes (packed 2-bit chunks for high bits)

    Returns:
        (low_bits_array, high_bits_bytes)
    """
    if len(indices) == 0:
        return np.array([], dtype=np.uint16), b""

    # 1. Low 16 bits -> uint16
    low_bits = (indices & 0xFFFF).astype(np.uint16)

    # 2. High 2 bits -> packed uint8
    high_parts = (indices >> 16) & 0x03

    # Pad to multiple of 4 for packing
    rem = len(high_parts) % 4
    if rem != 0:
        pad_len = 4 - rem
        high_parts = np.pad(high_parts, (0, pad_len), constant_values=0)

    # Reshape to (N/4, 4) and pack
    # Pack 4 items into one byte: (d << 6) | (c << 4) | (b << 2) | a
    high_parts_reshaped = high_parts.reshape(-1, 4)
    packed_high = (
        (high_parts_reshaped[:, 0])
        | (high_parts_reshaped[:, 1] << 2)
        | (high_parts_reshaped[:, 2] << 4)
        | (high_parts_reshaped[:, 3] << 6)
    ).astype(np.uint8)

    return low_bits, packed_high.tobytes()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExtractionConfig:
    """Configuration for logit extraction."""

    model_id: str
    dataset_id: str
    hf_output_repo: str

    # Processing options
    dataset_split: str = "train"

    # Nucleus (top-p) sampling: extract top logits that accumulate to p probability
    # with a hard maximum of 100 elements
    top_p: float = 0.98
    top_p_max_elements: int = 100

    # Additional sampled logits from remaining vocabulary
    sampled_n: int = 24

    # Batch sizing based on token budget (length -> samples per batch)
    # More conservative defaults to prevent OOM on typical GPUs
    # Format: {max_seq_len: max_batch_size}
    batch_budget_thresholds: dict = field(
        default_factory=lambda: {
            8192: 1,  # Very long sequences: single sample
            4096: 1,  # Long sequences: single sample
            2048: 2,  # Medium-long: batch of 2
            1024: 4,  # Medium: batch of 4
            512: 8,  # Short: batch of 8
            256: 16,  # Very short: batch of 16
        }
    )

    # Upload frequency
    upload_every_n_samples: int = 500

    # Model loading
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # Random seed for reproducibility
    seed: int = 42

    # Dataset columns (customizable per dataset)
    user_column: str = "question"
    assistant_column: str = "answer"
    index_column: Optional[str] = None  # If None, use row index
    subject_column: Optional[str] = None  # For stratified sampling

    # Pre-computed lengths (avoids re-tokenization for sorting)
    length_column: Optional[str] = None
    tokenized_dataset_id: Optional[str] = None

    # Sampling configuration (None = process all)
    # Example: {"": 20000, "*": 6000} - empty subject gets 20k, others get 6k
    sample_per_subject: Optional[dict] = None

    # Special token handling
    exclude_think_tags: bool = True
    exclude_end_tokens: bool = True

    # Processing limits
    limit_samples: Optional[int] = None
    start_offset: int = 0  # Skip first N samples
    max_seq_len: Optional[int] = None  # Skip samples longer than this

    # Resume options
    resume: bool = False
    force_restart: bool = False  # Ignore existing progress

    # Local cache for checkpoints
    cache_dir: str = ".logit_extraction_cache"

    # Custom chat template file
    chat_template_file: Optional[str] = None

    # Sampling Method
    random_sample: bool = False

    # Size-based upload: upload when buffer reaches this size (MB)
    # If None, uses upload_every_n_samples instead
    upload_size_mb: Optional[float] = None

    # Timed shutdown: stop after this many minutes and save unflushed data locally
    # On next run with --resume, pending local files will be uploaded first
    max_runtime_minutes: Optional[float] = None


# =============================================================================
# Streaming HuggingFace Upload
# =============================================================================


class StreamingHFUploader:
    """Handles incremental upload of processed samples to HuggingFace Hub."""

    def __init__(self, repo_id: str, config: ExtractionConfig, token: str = None):
        self.repo_id = repo_id
        self.config = config
        self.api = HfApi(token=token)
        self.buffer: list[dict] = []
        self.part_number = 0
        self.total_samples_uploaded = 0
        self.processed_indices: set[int] = set()
        self.buffer_size_bytes = 0  # Track buffer size for size-based flushing

        # Async upload support
        self.upload_executor = ThreadPoolExecutor(max_workers=1)
        self.pending_upload: Optional[tuple] = None  # (future, num_samples, filename)
        self._upload_lock = threading.Lock()

        # Ensure cache dir exists
        os.makedirs(config.cache_dir, exist_ok=True)

        # Pending directory for unflushed data saved on timed shutdown
        self.pending_dir = os.path.join(config.cache_dir, "pending")

        # Create repo if it doesn't exist
        try:
            self.api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"Note: Could not create repo (may already exist): {e}")

    def _local_checkpoint_path(self) -> str:
        return os.path.join(self.config.cache_dir, "checkpoint.json")

    def add_sample(self, sample: dict):
        """Add a processed sample to the buffer."""
        self.buffer.append(sample)
        self.processed_indices.add(int(sample["index"]))
        # Estimate sample size for size-based flushing
        self.buffer_size_bytes += self._estimate_sample_size(sample)

    def _estimate_sample_size(self, sample: dict) -> int:
        """Estimate the size of a sample in bytes."""
        size = 0
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                size += value.nbytes
            elif isinstance(value, bytes):
                size += len(value)
            elif isinstance(value, list):
                # Estimate list of floats/ints
                size += len(value) * 4
            else:
                size += 8  # Rough estimate for scalars
        return size

    def should_flush(self) -> bool:
        """Check if buffer should be flushed based on sample count or size."""
        # Size-based flush takes precedence if configured
        if self.config.upload_size_mb is not None:
            size_mb = self.buffer_size_bytes / (1024 * 1024)
            return size_mb >= self.config.upload_size_mb
        # Fall back to sample count
        return len(self.buffer) >= self.config.upload_every_n_samples

    def get_buffer_size_mb(self) -> float:
        """Get current buffer size in MB."""
        return self.buffer_size_bytes / (1024 * 1024)

    def _get_schema(self) -> pa.Schema:
        """Get the PyArrow schema for parquet files."""
        return pa.schema(
            [
                ("index", pa.int32()),
                ("num_tokens", pa.int32()),
                # Nucleus (flattened)
                ("top_indices_low", pa.list_(pa.uint16())),
                ("top_indices_high", pa.binary()),  # Flattened binary
                ("top_logits_quantized", pa.list_(pa.uint16())),
                ("top_counts", pa.list_(pa.uint8())),  # New counts column
                ("top_min", pa.list_(pa.float32())),
                ("top_max", pa.list_(pa.float32())),
                # Sampled (flattened)
                ("sampled_indices_low", pa.list_(pa.uint16())),
                ("sampled_indices_high", pa.binary()),
                ("sampled_logits_quantized", pa.list_(pa.uint16())),
                ("sampled_min", pa.list_(pa.float32())),
                ("sampled_max", pa.list_(pa.float32())),
                ("logsumexp", pa.list_(pa.float32())),
            ]
        )

    def _buffer_to_parquet_bytes(self) -> io.BytesIO:
        """Convert current buffer to parquet bytes."""
        df = pd.DataFrame(self.buffer)
        schema = self._get_schema()

        if self.part_number == 0:
            print("\n📊 PyArrow Schema:")
            print(schema)

        try:
            table = pa.Table.from_pandas(df, schema=schema)
        except Exception as e:
            print(f"\n❌ Error converting to PyArrow table: {e}")
            if len(self.buffer) > 0:
                print("First row types:")
                for k, v in self.buffer[0].items():
                    print(f"  {k}: {type(v)}")
            raise e

        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        return buf

    def _do_upload(self, buf: io.BytesIO, filename: str, num_samples: int, buffer_size_mb: float) -> bool:
        """Perform the actual upload (runs in background thread)."""
        try:
            self.api.upload_file(
                path_or_fileobj=buf,
                path_in_repo=filename,
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            with self._upload_lock:
                self.total_samples_uploaded += num_samples
            print(
                f"\n📤 Uploaded {filename} ({num_samples} samples, "
                f"{buffer_size_mb:.1f}MB, total: {self.total_samples_uploaded})"
            )
            return True
        except Exception as e:
            print(f"\n⚠️ Upload failed for {filename}: {e}")
            # Save locally as backup
            backup_path = os.path.join(
                self.config.cache_dir, f"backup_{filename.replace('/', '_')}"
            )
            with open(backup_path, "wb") as f:
                buf.seek(0)
                f.write(buf.read())
            print(f"   Saved backup to {backup_path}")
            return False

    def _wait_for_pending_upload(self):
        """Wait for any pending upload to complete."""
        if self.pending_upload is not None:
            future, num_samples, filename = self.pending_upload
            try:
                future.result()  # Block until upload completes
            except Exception as e:
                print(f"\n⚠️ Pending upload error: {e}")
            self.pending_upload = None
            self._save_checkpoint()

    def flush(self, final: bool = False):
        """Upload buffered samples as a Parquet file to HuggingFace (async unless final)."""
        # Wait for any previous upload to complete first
        self._wait_for_pending_upload()

        if not self.buffer:
            return

        buf = self._buffer_to_parquet_bytes()
        buffer_size_mb = self.buffer_size_bytes / (1024 * 1024)
        filename = f"data/part-{self.part_number:05d}.parquet"
        num_samples = len(self.buffer)

        # Track indices before clearing buffer
        for sample in self.buffer:
            self.processed_indices.add(int(sample["index"]))

        # Clear buffer immediately (data is in buf now)
        self.buffer = []
        self.buffer_size_bytes = 0
        self.part_number += 1

        if final:
            # Synchronous upload for final flush
            self._do_upload(buf, filename, num_samples, buffer_size_mb)
            self._save_checkpoint()
        else:
            # Async upload - submit to executor and continue processing
            future = self.upload_executor.submit(
                self._do_upload, buf, filename, num_samples, buffer_size_mb
            )
            self.pending_upload = (future, num_samples, filename)

    def is_uploading(self) -> bool:
        """Check if an upload is currently in progress."""
        return self.pending_upload is not None

    def shutdown(self):
        """Shutdown the uploader, waiting for pending uploads."""
        self._wait_for_pending_upload()
        self.upload_executor.shutdown(wait=True)

    def save_pending_locally(self):
        """Save unflushed buffer locally for later upload (used on timed shutdown)."""
        # Wait for any in-flight upload to complete first
        self._wait_for_pending_upload()

        if not self.buffer:
            return None

        os.makedirs(self.pending_dir, exist_ok=True)
        buf = self._buffer_to_parquet_bytes()
        buffer_size_mb = self.buffer_size_bytes / (1024 * 1024)

        # Save with timestamp to avoid conflicts
        pending_file = os.path.join(self.pending_dir, f"pending-{self.part_number:05d}.parquet")
        with open(pending_file, "wb") as f:
            f.write(buf.read())

        print(
            f"\n💾 Saved pending data locally: {pending_file} "
            f"({len(self.buffer)} samples, {buffer_size_mb:.1f}MB)"
        )

        # Save checkpoint with pending info
        self._save_checkpoint()

        self.buffer = []
        self.buffer_size_bytes = 0
        return pending_file

    def load_pending_into_buffer(self) -> int:
        """Load any pending local files back into the buffer for merging with new data.

        Pending files are deleted after being loaded. The data will be uploaded
        together with new samples when the buffer reaches the upload threshold.

        Returns:
            Number of samples loaded from pending files.
        """
        if not os.path.exists(self.pending_dir):
            return 0

        pending_files = sorted(glob.glob(os.path.join(self.pending_dir, "*.parquet")))
        if not pending_files:
            return 0

        print(f"\n📂 Loading {len(pending_files)} pending local file(s) into buffer...")
        total_loaded = 0

        for pending_file in pending_files:
            try:
                # Read the parquet file
                table = pq.read_table(pending_file)
                df = table.to_pandas()

                # Convert each row back to the sample dict format
                for _, row in df.iterrows():
                    sample = {
                        "index": row["index"],
                        "num_tokens": row["num_tokens"],
                        "top_indices_low": row["top_indices_low"],
                        "top_indices_high": row["top_indices_high"],
                        "top_logits_quantized": row["top_logits_quantized"],
                        "top_counts": row["top_counts"],
                        "top_min": row["top_min"],
                        "top_max": row["top_max"],
                        "sampled_indices_low": row["sampled_indices_low"],
                        "sampled_indices_high": row["sampled_indices_high"],
                        "sampled_logits_quantized": row["sampled_logits_quantized"],
                        "sampled_min": row["sampled_min"],
                        "sampled_max": row["sampled_max"],
                        "logsumexp": row["logsumexp"],
                    }
                    self.buffer.append(sample)
                    self.processed_indices.add(int(sample["index"]))
                    self.buffer_size_bytes += self._estimate_sample_size(sample)

                total_loaded += len(df)

                # Delete local file after loading
                os.remove(pending_file)
                print(f"   ✅ Loaded {pending_file} ({len(df)} samples)")

            except Exception as e:
                print(f"   ⚠️ Failed to load {pending_file}: {e}")
                print(f"      File will remain for manual inspection")

        # Clean up empty pending directory
        if os.path.exists(self.pending_dir) and not os.listdir(self.pending_dir):
            os.rmdir(self.pending_dir)

        if total_loaded > 0:
            print(
                f"   📂 Loaded {total_loaded} samples into buffer "
                f"({self.get_buffer_size_mb():.1f}MB)"
            )

        return total_loaded

    def _save_checkpoint(self):
        """Save checkpoint metadata locally and to HuggingFace."""
        checkpoint = {
            "part_number": self.part_number,
            "total_samples": self.total_samples_uploaded,
            "processed_indices": sorted(list(self.processed_indices)),
            "config": {
                "model_id": self.config.model_id,
                "dataset_id": self.config.dataset_id,
                "top_p": self.config.top_p,
                "top_p_max_elements": self.config.top_p_max_elements,
                "sampled_n": self.config.sampled_n,
            },
        }

        # Save local checkpoint
        local_path = self._local_checkpoint_path()
        with open(local_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        # Save remote checkpoint
        buf = io.BytesIO(json.dumps(checkpoint, indent=2).encode())
        try:
            self.api.upload_file(
                path_or_fileobj=buf,
                path_in_repo="checkpoint.json",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
        except Exception:
            pass  # Non-critical

    def get_resume_info(self) -> tuple[int, set[int], int]:
        """
        Get resume information from local checkpoint and HuggingFace dataset.
        Also loads any pending local files from a previous timed shutdown into the buffer.

        Returns:
            (part_number, set of processed indices, total existing samples)
        """
        processed_indices: set[int] = set()
        part_number = 0
        total_existing = 0

        # First, try to load local checkpoint (fastest)
        local_path = self._local_checkpoint_path()
        if os.path.exists(local_path) and not self.config.force_restart:
            try:
                with open(local_path) as f:
                    checkpoint = json.load(f)
                    part_number = checkpoint.get("part_number", 0)
                    processed_indices = set(checkpoint.get("processed_indices", []))
                    print(
                        f"📍 Loaded local checkpoint: {len(processed_indices)} samples"
                    )
            except Exception as e:
                print(f"Note: Could not load local checkpoint: {e}")

        # Load any pending local files into buffer (will be uploaded with next chunk)
        self.load_pending_into_buffer()

        # Verify against HuggingFace dataset
        try:
            # OPTIMIZATION: Enable prefetching and larger buffer for faster streaming
            fragment_scan_options = pa_dataset.ParquetFragmentScanOptions(
                cache_options=pa.CacheOptions(
                    prefetch_limit=1,  # Prefetch next chunk while processing current
                    range_size_limit=128 << 20,  # 128 MiB block size (default is 32MiB)
                ),
            )

            # OPTIMIZATION: Use columns= directly (faster than .select_columns())
            ds = load_dataset(
                self.repo_id,
                split="train",
                streaming=True,
                columns=["index"],
                fragment_scan_options=fragment_scan_options,
            )

            hf_indices = set()
            # OPTIMIZATION: Batch iteration instead of row-by-row (~10x faster)
            for batch in tqdm(ds.batch(batch_size=10000), desc="Verifying HF dataset"):
                hf_indices.update(batch["index"])
                total_existing += len(batch["index"])

            # Use HF indices as source of truth, but preserve local additions
            if hf_indices and not self.config.force_restart:
                # Keep indices from both sources
                processed_indices = processed_indices | hf_indices
                print(f"📍 HF dataset has {len(hf_indices)} samples")
        except Exception as e:
            if total_existing == 0:
                print(f"Note: Could not load existing HF data: {e}")

        self.part_number = part_number
        self.processed_indices = processed_indices
        self.total_samples_uploaded = len(processed_indices)

        return part_number, processed_indices, total_existing


# =============================================================================
# Graceful Interruption Handler
# =============================================================================


class GracefulInterrupt:
    """Handle Ctrl+C gracefully by flushing data before exit."""

    def __init__(self, uploader: StreamingHFUploader):
        self.interrupted = False
        self.uploader = uploader
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        if self.interrupted:
            print("\n🛑 Force quit - some data may be lost!")
            sys.exit(1)
        print("\n\n⏸️  Interrupt received. Flushing data and exiting gracefully...")
        print("   (Press Ctrl+C again to force quit)")
        self.interrupted = True

    def check_and_exit(self):
        """Check if we should stop processing and exit gracefully."""
        if self.interrupted:
            self.uploader.flush(final=True)
            print(
                f"\n✅ Saved progress. {self.uploader.total_samples_uploaded} samples uploaded."
            )
            print(f"   Run with --resume to continue from where you left off.")
            sys.exit(0)


class TimedShutdown:
    """Handle automatic shutdown after a specified runtime."""

    def __init__(
        self,
        uploader: StreamingHFUploader,
        max_runtime_minutes: Optional[float] = None,
    ):
        self.uploader = uploader
        self.max_runtime_minutes = max_runtime_minutes
        self.start_time = time.time()
        self.should_stop = False

        if max_runtime_minutes:
            print(f"⏱️  Timed shutdown enabled: will stop after {max_runtime_minutes:.1f} minutes")

    def check_timeout(self) -> bool:
        """Check if we've exceeded the time limit. Returns True if should stop."""
        if self.max_runtime_minutes is None:
            return False

        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes >= self.max_runtime_minutes:
            self.should_stop = True
            return True
        return False

    def get_elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        return (time.time() - self.start_time) / 60

    def get_remaining_minutes(self) -> Optional[float]:
        """Get remaining time in minutes, or None if no limit."""
        if self.max_runtime_minutes is None:
            return None
        elapsed = self.get_elapsed_minutes()
        return max(0, self.max_runtime_minutes - elapsed)

    def shutdown_gracefully(self):
        """Perform graceful shutdown, saving unflushed data locally."""
        print(f"\n\n⏱️  Time limit reached ({self.max_runtime_minutes:.1f} minutes).")
        print("   Saving unflushed data locally...")

        # Save any unflushed data to pending directory
        pending_file = self.uploader.save_pending_locally()

        print(
            f"\n✅ Saved progress. {self.uploader.total_samples_uploaded} samples uploaded to HF."
        )
        if pending_file:
            print(f"   Pending data saved to: {pending_file}")
        print(f"   Run with --resume to continue (pending data will be uploaded first)")


# =============================================================================
# Dataset and DataLoader
# =============================================================================


class ConversationDataset(Dataset):
    """Dataset that wraps conversations for logit extraction."""

    def __init__(self, df: pd.DataFrame, config: ExtractionConfig):
        self.df = df.reset_index(drop=True)
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        messages = [
            {"role": "user", "content": row[self.config.user_column]},
            {"role": "assistant", "content": row[self.config.assistant_column]},
        ]

        # Get original index
        if self.config.index_column and self.config.index_column in row:
            original_idx = int(row[self.config.index_column])
        elif "original_index" in row:
            original_idx = int(row["original_index"])
        else:
            original_idx = idx

        # Get length if available
        length = row.get("length", 0)

        return messages, length, original_idx


class TokenBudgetBatchSampler(Sampler):
    """Batch sampler that groups by token budget to maximize GPU utilization.

    The thresholds dict maps max sequence length -> max batch size.
    Examples:
        - {512: 8, 1024: 4} means:
          - Sequences <= 512 tokens: batch size 8
          - Sequences <= 1024 tokens: batch size 4
          - Sequences > 1024 tokens: batch size 1 (default)

    The DataFrame should be sorted by length descending for this to work efficiently.
    """

    def __init__(self, lengths: list[int], thresholds: dict, drop_last: bool = False):
        self.lengths = lengths
        self.indices = list(range(len(lengths)))
        # Sort thresholds by length ASCENDING so we match the smallest fitting bucket first
        self.thresholds = sorted(thresholds.items(), key=lambda x: x[0])
        self.drop_last = drop_last
        # Pre-compute number of batches for __len__
        self._num_batches = self._compute_num_batches()

    def _get_max_batch_size(self, length: int) -> int:
        """Get max batch size for a given sequence length.

        Thresholds are sorted ascending. We find the smallest threshold >= length.
        E.g., for thresholds {256: 16, 512: 8, 1024: 4}:
          - length 200 -> matches 256 -> batch size 16
          - length 500 -> matches 512 -> batch size 8
          - length 2000 -> no match -> batch size 1
        """
        for thr, cnt in self.thresholds:
            if length <= thr:
                return cnt
        return 1  # Default for very long sequences (longer than any threshold)

    def _compute_num_batches(self) -> int:
        """Compute total number of batches."""
        if not self.lengths:
            return 0

        num_batches = 0
        batch_size = 0
        current_max = float("inf")

        for idx in self.indices:
            length = self.lengths[idx]
            allowed = self._get_max_batch_size(length)

            # Start new batch if current batch is full
            if batch_size >= current_max:
                num_batches += 1
                batch_size = 0
                current_max = allowed

            batch_size += 1
            current_max = min(current_max, allowed)

        if batch_size > 0:
            num_batches += 1

        return num_batches

    def __iter__(self):
        batch = []
        current_max = float("inf")

        for idx in self.indices:
            length = self.lengths[idx]
            allowed = self._get_max_batch_size(length)

            # Start new batch if current batch is full or max batch size decreased
            if len(batch) >= current_max:
                yield batch
                batch = []
                current_max = allowed

            batch.append(idx)
            current_max = min(current_max, allowed)

        if batch and (not self.drop_last or len(batch) == current_max):
            yield batch

    def __len__(self):
        return self._num_batches


# =============================================================================
# Assistant Mask Verification
# =============================================================================


def verify_assistant_mask(
    tokenizer, sample_conversation: list[dict], chat_template: Optional[str] = None
) -> dict:
    """
    Verify the assistant mask correctly identifies generated tokens.

    This is crucial for ensuring we only extract logits for assistant-generated
    tokens, not for the user prompt or template tokens.

    Args:
        tokenizer: The tokenizer to test
        sample_conversation: A sample conversation to verify
        chat_template: Optional custom chat template to use

    Returns:
        Tokenization result with verification info
    """
    result = tokenizer.apply_chat_template(
        sample_conversation,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
        chat_template=chat_template,
    )

    input_ids = result["input_ids"].squeeze()
    mask = result["assistant_masks"].squeeze().bool()

    assistant_ids = input_ids[mask]
    non_assistant_ids = input_ids[~mask]

    print("=" * 60)
    print("ASSISTANT MASK VERIFICATION")
    print("=" * 60)
    print(f"Total tokens: {input_ids.numel()}")
    print(f"Assistant tokens: {assistant_ids.numel()}")
    print(f"Non-assistant tokens: {non_assistant_ids.numel()}")
    print()

    assistant_text = tokenizer.decode(assistant_ids)
    non_assistant_text = tokenizer.decode(non_assistant_ids)

    print("--- Assistant text (should be the response) ---")
    print(assistant_text[:500] + "..." if len(assistant_text) > 500 else assistant_text)
    print()
    print("--- Non-assistant text (should be prompt/template) ---")
    print(
        non_assistant_text[:500] + "..."
        if len(non_assistant_text) > 500
        else non_assistant_text
    )
    print("=" * 60)

    # Sanity checks
    expected_response = sample_conversation[-1]["content"]
    if expected_response.strip() in assistant_text.strip():
        print("✅ Assistant mask correctly captures the assistant response")
    else:
        print(
            "⚠️ Warning: Assistant response may not be fully captured in assistant tokens"
        )

    return result


# =============================================================================
# Logit Extraction Logic
# =============================================================================

UINT16_MAX = 65535


def create_collate_fn(tokenizer, chat_template: Optional[str] = None):
    """Create a collate function for the DataLoader."""

    def collate_fn(batch):
        messages_list = [b[0] for b in batch]
        indexes = np.array([b[2] for b in batch], dtype=np.int64)

        tokenized = tokenizer.apply_chat_template(
            messages_list,
            return_dict=True,
            padding="longest",
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            chat_template=chat_template,
        )

        return tokenized, indexes

    return collate_fn


@dataclass
class GPUBatchResult:
    """Intermediate GPU results for async CPU post-processing."""
    # Nucleus data (on CPU after transfer)
    nucleus_indices: np.ndarray  # (total_tokens, top_p_max) padded
    nucleus_logits: np.ndarray   # (total_tokens, top_p_max) padded
    nucleus_counts: np.ndarray   # (total_tokens,) actual count per token
    nucleus_min: np.ndarray      # (total_tokens,)
    nucleus_max: np.ndarray      # (total_tokens,)
    # Sampled data (on CPU after transfer)
    sampled_indices: np.ndarray  # (total_tokens, sampled_n)
    sampled_logits: np.ndarray   # (total_tokens, sampled_n) quantized
    sampled_min: np.ndarray      # (total_tokens,)
    sampled_max: np.ndarray      # (total_tokens,)
    # Other
    logsumexp: np.ndarray        # (total_tokens,)
    sizes: list[int]             # tokens per sample in batch
    indexes: np.ndarray          # original dataset indices


def gpu_process_batch(
    model,
    batch: dict,
    indexes: np.ndarray,
    config: ExtractionConfig,
    exclude_tokens: Optional[torch.Tensor] = None,
    think_token_ids: Optional[torch.Tensor] = None,
) -> tuple[Optional[GPUBatchResult], BatchTimingMetrics]:
    """
    GPU-optimized batch processing. All heavy computation on GPU, single CPU transfer.

    Returns:
        (GPUBatchResult for CPU post-processing, timing metrics)
    """
    device = next(model.parameters()).device
    metrics = BatchTimingMetrics()

    # Start timing model forward pass
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_start = time.perf_counter()

    with torch.no_grad():
        # Forward pass to get hidden states
        outputs = model.model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state

        # Build extraction mask from assistant_masks
        assistant_mask = batch["assistant_masks"].bool().to(device)

        # Exclude end-of-generation tokens if configured
        if config.exclude_end_tokens and exclude_tokens is not None:
            exclude_tokens = exclude_tokens.to(device)
            is_not_after_special = ~torch.isin(
                batch["input_ids"].to(device).roll(1), exclude_tokens
            )
            assistant_mask = assistant_mask & is_not_after_special

        # Exclude thinking tag positions if configured
        if config.exclude_think_tags and think_token_ids is not None:
            think_token_ids = think_token_ids.to(device)
            input_ids_device = batch["input_ids"].to(device)
            is_not_think = torch.ones_like(input_ids_device, dtype=torch.bool)
            for think_id in think_token_ids:
                is_not_think &= ~torch.isin(input_ids_device, think_id)
                for shift in range(1, 4):
                    is_not_think &= ~torch.isin(input_ids_device.roll(shift), think_id)
            assistant_mask = assistant_mask & is_not_think

        # Get logits for assistant positions only
        extract_positions = assistant_mask.nonzero(as_tuple=True)
        N = extract_positions[0].numel()
        if N == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            metrics.model_time_ms = (time.perf_counter() - model_start) * 1000
            return None, metrics

        logits = model.lm_head(hidden_states[extract_positions])  # (N, vocab)
        metrics.num_tokens = N

        # Compute logsumexp for normalization info
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1)
        probs = torch.exp(logits_f - lse.unsqueeze(-1))

        # Count tokens per sample in batch
        sizes = assistant_mask.sum(dim=-1).tolist()

        # =====================================================================
        # VECTORIZED NUCLEUS EXTRACTION (no Python loops!)
        # =====================================================================
        vocab_size = probs.size(-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)

        # Compute cumulative sum
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find nucleus mask: cumsum <= top_p, with position limit
        nucleus_mask = cumsum_probs <= config.top_p
        nucleus_mask[:, 0] = True  # Always include top token

        # Apply hard maximum constraint
        position_indices = torch.arange(vocab_size, device=device)
        position_mask = position_indices < config.top_p_max_elements
        nucleus_mask = nucleus_mask & position_mask

        # Count elements in nucleus per row (vectorized)
        nucleus_counts = nucleus_mask.sum(dim=-1)  # (N,)

        # Extract top-k elements where k = top_p_max_elements (padded)
        # This is much faster than variable-length extraction
        k = config.top_p_max_elements
        top_indices = sorted_idx[:, :k]  # (N, k) - original vocab indices
        top_logits = logits_f.gather(-1, top_indices)  # (N, k)

        # Quantize nucleus logits on GPU (vectorized per-row)
        top_min = top_logits.min(dim=-1).values  # (N,)
        top_max = top_logits.max(dim=-1).values  # (N,)
        top_range = (top_max - top_min).clamp(min=1e-10)
        top_scaled = ((top_logits - top_min.unsqueeze(-1)) / top_range.unsqueeze(-1) * UINT16_MAX).round()
        top_quantized = top_scaled.clamp(0, UINT16_MAX).to(torch.int32)

        # =====================================================================
        # VECTORIZED SAMPLING FROM REMAINING VOCAB
        # =====================================================================
        # Create mask in vocab order using scatter (vectorized, no loop!)
        nucleus_mask_vocab = torch.zeros(N, vocab_size, dtype=torch.bool, device=device)
        # Create row indices for scatter
        row_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)
        # Only mark positions within actual nucleus count
        col_mask = torch.arange(k, device=device).unsqueeze(0) < nucleus_counts.unsqueeze(1)
        # Use advanced indexing to set True values
        valid_rows = row_idx[col_mask]
        valid_cols = top_indices[col_mask]
        nucleus_mask_vocab[valid_rows, valid_cols] = True

        # Prepare sampling distribution
        probs_rest = probs.clone()
        probs_rest[nucleus_mask_vocab] = 0.0
        rest_sum = probs_rest.sum(dim=-1, keepdim=True)

        # Handle edge case: all probability in nucleus
        zero_rest = rest_sum.squeeze(-1) < 1e-10
        if zero_rest.any():
            uniform_mask = ~nucleus_mask_vocab[zero_rest]
            counts = uniform_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
            probs_rest[zero_rest] = uniform_mask.float() / counts
            rest_sum = probs_rest.sum(dim=-1, keepdim=True)

        # Renormalize
        probs_rest = probs_rest / rest_sum.clamp(min=1e-30)

        # Sample from remaining distribution
        sampled_idx = torch.multinomial(probs_rest, config.sampled_n)
        sampled_logits = logits_f.gather(-1, sampled_idx)

        # Quantize sampled logits on GPU
        samp_min = sampled_logits.min(dim=-1).values
        samp_max = sampled_logits.max(dim=-1).values
        samp_range = (samp_max - samp_min).clamp(min=1e-10)
        samp_scaled = ((sampled_logits - samp_min.unsqueeze(-1)) / samp_range.unsqueeze(-1) * UINT16_MAX).round()
        samp_quantized = samp_scaled.clamp(0, UINT16_MAX).to(torch.int32)

        # Synchronize and record model time
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics.model_time_ms = (time.perf_counter() - model_start) * 1000

        # =====================================================================
        # SINGLE BATCHED CPU TRANSFER
        # =====================================================================
        # Transfer all tensors to CPU in one batch (more efficient)
        result = GPUBatchResult(
            nucleus_indices=top_indices.cpu().numpy().astype(np.int32),
            nucleus_logits=top_quantized.cpu().numpy().astype(np.uint16),
            nucleus_counts=nucleus_counts.cpu().numpy().astype(np.uint8),
            nucleus_min=top_min.cpu().numpy().astype(np.float32),
            nucleus_max=top_max.cpu().numpy().astype(np.float32),
            sampled_indices=sampled_idx.cpu().numpy().astype(np.int32),
            sampled_logits=samp_quantized.cpu().numpy().astype(np.uint16),
            sampled_min=samp_min.cpu().numpy().astype(np.float32),
            sampled_max=samp_max.cpu().numpy().astype(np.float32),
            logsumexp=lse.cpu().numpy().astype(np.float32),
            sizes=sizes,
            indexes=indexes,
        )

    return result, metrics


def cpu_postprocess_batch(gpu_result: GPUBatchResult, config: ExtractionConfig) -> list[dict]:
    """
    CPU post-processing: pack indices and create final sample dicts.

    This runs on CPU and can be overlapped with GPU processing of next batch.
    """
    results = []
    offset = 0

    for i, size in enumerate(gpu_result.sizes):
        if size == 0:
            continue

        end = offset + size

        # Extract this sample's nucleus data
        sample_nucleus_idx = gpu_result.nucleus_indices[offset:end]  # (size, k)
        sample_nucleus_quant = gpu_result.nucleus_logits[offset:end]  # (size, k)
        sample_nucleus_counts = gpu_result.nucleus_counts[offset:end]  # (size,)
        sample_nucleus_min = gpu_result.nucleus_min[offset:end]  # (size,)
        sample_nucleus_max = gpu_result.nucleus_max[offset:end]  # (size,)

        # Flatten only valid nucleus elements (using counts)
        flat_nucleus_idx_parts = []
        flat_nucleus_quant_parts = []
        for j in range(size):
            cnt = sample_nucleus_counts[j]
            flat_nucleus_idx_parts.append(sample_nucleus_idx[j, :cnt])
            flat_nucleus_quant_parts.append(sample_nucleus_quant[j, :cnt])

        if flat_nucleus_idx_parts:
            flat_nucleus_idx = np.concatenate(flat_nucleus_idx_parts)
            flat_nucleus_quant = np.concatenate(flat_nucleus_quant_parts)
            nucleus_low, nucleus_high_bytes = pack_indices(flat_nucleus_idx)
        else:
            nucleus_low = np.array([], dtype=np.uint16)
            nucleus_high_bytes = b""
            flat_nucleus_quant = np.array([], dtype=np.uint16)

        # Extract this sample's sampled data
        sample_samp_idx = gpu_result.sampled_indices[offset:end]  # (size, sampled_n)
        sample_samp_quant = gpu_result.sampled_logits[offset:end]
        sample_samp_min = gpu_result.sampled_min[offset:end]
        sample_samp_max = gpu_result.sampled_max[offset:end]

        # Flatten sampled data
        flat_samp_idx = sample_samp_idx.flatten()
        flat_samp_quant = sample_samp_quant.flatten()
        samp_low, samp_high_bytes = pack_indices(flat_samp_idx)

        results.append({
            "index": np.int32(gpu_result.indexes[i]),
            "num_tokens": np.int32(size),
            # Nucleus (flattened)
            "top_indices_low": nucleus_low,
            "top_indices_high": nucleus_high_bytes,
            "top_logits_quantized": flat_nucleus_quant,
            "top_counts": sample_nucleus_counts,
            "top_min": sample_nucleus_min.tolist(),
            "top_max": sample_nucleus_max.tolist(),
            # Sampled (flattened)
            "sampled_indices_low": samp_low,
            "sampled_indices_high": samp_high_bytes,
            "sampled_logits_quantized": flat_samp_quant,
            "sampled_min": sample_samp_min.tolist(),
            "sampled_max": sample_samp_max.tolist(),
            "logsumexp": gpu_result.logsumexp[offset:end].tolist(),
        })
        offset = end

    return results


class AsyncPostProcessor:
    """Handles async CPU post-processing while GPU works on next batch."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_future = None
        self.pending_metrics: Optional[BatchTimingMetrics] = None

    def submit(self, gpu_result: GPUBatchResult, config: ExtractionConfig, metrics: BatchTimingMetrics):
        """Submit GPU result for async CPU post-processing."""
        # Wait for any previous pending work first
        results = self.get_results()

        # Submit new work
        self.pending_future = self.executor.submit(
            self._process_with_timing, gpu_result, config, metrics
        )
        self.pending_metrics = metrics

        return results

    def _process_with_timing(
        self, gpu_result: GPUBatchResult, config: ExtractionConfig, metrics: BatchTimingMetrics
    ) -> tuple[list[dict], BatchTimingMetrics]:
        """Process and record timing."""
        start = time.perf_counter()
        results = cpu_postprocess_batch(gpu_result, config)
        metrics.postprocess_time_ms = (time.perf_counter() - start) * 1000
        return results, metrics

    def get_results(self) -> Optional[tuple[list[dict], BatchTimingMetrics]]:
        """Get results from pending work, if any."""
        if self.pending_future is None:
            return None

        results, metrics = self.pending_future.result()
        self.pending_future = None
        self.pending_metrics = None
        return results, metrics

    def flush(self) -> Optional[tuple[list[dict], BatchTimingMetrics]]:
        """Flush any remaining pending work."""
        return self.get_results()

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


def process_batch(
    model,
    tokenizer,
    batch: dict,
    indexes: np.ndarray,
    config: ExtractionConfig,
    exclude_tokens: Optional[torch.Tensor] = None,
    think_token_ids: Optional[torch.Tensor] = None,
) -> Optional[list[dict]]:
    """
    Legacy synchronous batch processing (for backwards compatibility).
    """
    gpu_result, metrics = gpu_process_batch(
        model, batch, indexes, config, exclude_tokens, think_token_ids
    )
    if gpu_result is None:
        return None
    return cpu_postprocess_batch(gpu_result, config)


# =============================================================================
# Dataset Preparation
# =============================================================================


def prepare_dataset(
    config: ExtractionConfig, skip_indices: set[int]
) -> tuple[pd.DataFrame, list[int]]:
    """
    Load and prepare the dataset for extraction.

    Args:
        config: Extraction configuration
        skip_indices: Set of indices to skip (already processed)

    Returns:
        (DataFrame, list of lengths)
    """
    print(f"📦 Loading dataset: {config.dataset_id}")
    dataset = load_dataset(config.dataset_id, split=config.dataset_split)

    # Convert to DataFrame and add original index
    df = dataset.to_pandas()
    df["original_index"] = df.index

    # Load pre-computed lengths if available
    if config.tokenized_dataset_id and config.length_column:
        print(f"📏 Loading lengths from: {config.tokenized_dataset_id}")
        length_ds = load_dataset(
            config.tokenized_dataset_id, split=config.dataset_split
        )
        df["length"] = length_ds[config.length_column]
    elif config.length_column and config.length_column in df.columns:
        df["length"] = df[config.length_column]
    else:
        # Mark for length computation during iteration
        df["length"] = 0

    # Apply stratified sampling/limiting if configured
    if config.subject_column:
        if config.subject_column in df.columns:
            print(f"🎲 Applying group-based limits by {config.subject_column}")

            # Determine limits per group
            # If sample_per_subject is provided, use it.
            # Otherwise if limit_samples is provided, use it as limit PER GROUP.

            def apply_group_limit(group):
                subject = group.name
                limit = None

                if config.sample_per_subject:
                    limit = config.sample_per_subject.get(
                        subject, config.sample_per_subject.get("*", None)
                    )

                if limit is None and config.limit_samples:
                    limit = config.limit_samples

                if limit is not None and len(group) > limit:
                    if config.random_sample:
                        # Use reproducible random sampling
                        rng = np.random.default_rng(config.seed)
                        # Simple random selection of indices from this group
                        indices = rng.choice(len(group), size=limit, replace=False)
                        return group.iloc[indices]
                    else:
                        return group.head(limit)
                return group

            # Apply limits
            df = df.groupby(config.subject_column, group_keys=False).apply(
                apply_group_limit, include_groups=True
            )

            # Report counts per group
            print("\n📊 Samples per group:")
            counts = df[config.subject_column].value_counts().sort_index()
            for subject, count in counts.items():
                print(f"   - {subject}: {count}")
            print(f"   Total: {len(df)}\n")

        else:
            print(
                f"⚠️ Subject column '{config.subject_column}' not found, skipping group limits"
            )
            # Fallback to global limit if subject col missing
            if config.limit_samples:
                df = df.head(config.limit_samples)
                print(f"🔢 Limiting to {len(df)} samples (global)")

    # Filter out already processed indices
    if skip_indices:
        original_len = len(df)
        df = df[~df["original_index"].isin(skip_indices)]
        print(f"⏭️  Skipping {original_len - len(df)} already processed samples")

    # Apply start offset (only if not doing group limits, or applied after?)
    # Usually start offset is for global restart.
    if config.start_offset > 0:
        df = df.iloc[config.start_offset :]
        print(f"⏭️  Skipped first {config.start_offset} samples")

    # Limit global samples if NO subject column was used (or if explicit global limit needed?)
    # The requirement says "limit applies to the number of examples per group".
    # If subject column IS NOT present, we use limit_samples as global limit.
    if config.limit_samples and not config.subject_column:
        if len(df) > config.limit_samples:
            if config.random_sample:
                print(f"🎲 Randomly sampling {config.limit_samples} samples")
                df = df.sample(
                    n=config.limit_samples,
                    random_state=np.random.RandomState(config.seed),
                )
            else:
                df = df.head(config.limit_samples)
                print(f"🔢 Limiting to {len(df)} samples")
        else:
            print(f"🔢 Limiting to {len(df)} samples")

    # Filter out samples that are too long
    if config.max_seq_len and "length" in df.columns:
        original_len = len(df)
        df = df[df["length"] <= config.max_seq_len]
        skipped = original_len - len(df)
        if skipped > 0:
            print(
                f"⏭️  Skipped {skipped} samples longer than {config.max_seq_len} tokens"
            )

    # Sort by length descending for better batching
    if "length" in df.columns and df["length"].sum() > 0:
        df = df.sort_values("length", ascending=False)

    lengths = df["length"].tolist() if "length" in df.columns else [0] * len(df)

    print(f"📊 Total samples to process: {len(df)}")

    return df, lengths


# =============================================================================
# Main Extraction Pipeline
# =============================================================================


def run_extraction(config: ExtractionConfig, verify_mask: bool = False):
    """
    Run the full extraction pipeline.

    Args:
        config: Extraction configuration
        verify_mask: Whether to verify assistant mask on first sample
    """
    print("=" * 60)
    print("LOGIT EXTRACTION PIPELINE")
    print("=" * 60)
    print(f"Model: {config.model_id}")
    print(f"Dataset: {config.dataset_id}")
    print(f"Output: {config.hf_output_repo}")
    print(
        f"Top-P: {config.top_p}, Max Elements: {config.top_p_max_elements}, Sampled-N: {config.sampled_n}"
    )
    print("=" * 60)

    # Initialize uploader
    uploader = StreamingHFUploader(config.hf_output_repo, config)

    # Get resume info
    skip_indices: set[int] = set()
    if config.resume and not config.force_restart:
        print("\n🔄 Checking for existing progress...")
        _, skip_indices, existing_count = uploader.get_resume_info()
        if skip_indices:
            print(f"   Found {len(skip_indices)} already processed samples")
        if existing_count > 0:
            print(f"   HF dataset has {existing_count} samples")
    elif config.force_restart:
        print("\n🔄 Force restart - ignoring existing progress")

    # Set up graceful interrupt handler
    interrupt_handler = GracefulInterrupt(uploader)

    # Set up timed shutdown handler
    timed_shutdown = TimedShutdown(uploader, config.max_runtime_minutes)

    # Load model and tokenizer
    print(f"\n🤖 Loading model: {config.model_id}")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map=config.device,
        attn_implementation=config.attn_implementation,
        torch_dtype=dtype_map.get(config.torch_dtype, torch.bfloat16),
    )
    # CRITICAL: Set model to eval mode and disable gradient computation
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load custom chat template if provided
    custom_chat_template = None
    if config.chat_template_file:
        print(f"📝 Loading custom chat template from: {config.chat_template_file}")
        with open(config.chat_template_file, "r") as f:
            custom_chat_template = f.read()
        print("   ✅ Custom chat template loaded")

    # Get special tokens to exclude
    end_token_strs = ["<|im_end|>", "<|endoftext|>"]
    exclude_token_ids = []
    for s in end_token_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            exclude_token_ids.extend(ids)
    exclude_tokens = (
        torch.tensor(exclude_token_ids, dtype=torch.long) if exclude_token_ids else None
    )

    # Get think token IDs for exclusion
    think_token_ids = []
    for tag in ["<|im_start|>", "<|im_end|>", "</think>", "<|endoftext|>"]:
        ids = tokenizer.encode(tag, add_special_tokens=False)
        if ids:
            think_token_ids.extend(ids)
    think_tokens_tensor = (
        torch.tensor(think_token_ids, dtype=torch.long) if think_token_ids else None
    )

    # Prepare dataset
    df, lengths = prepare_dataset(config, skip_indices)

    if len(df) == 0:
        print("✅ Nothing to process - all samples already extracted!")
        return

    # Verify assistant mask if requested
    if verify_mask:
        sample_messages = [
            {"role": "user", "content": df.iloc[0][config.user_column]},
            {"role": "assistant", "content": df.iloc[0][config.assistant_column]},
        ]
        verify_assistant_mask(
            tokenizer, sample_messages, chat_template=custom_chat_template
        )
        input("\nPress Enter to continue with extraction...")

    # Create dataset and dataloader
    dataset = ConversationDataset(df, config)
    sampler = TokenBudgetBatchSampler(lengths, config.batch_budget_thresholds)
    collate_fn = create_collate_fn(tokenizer, chat_template=custom_chat_template)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 to avoid issues with multiprocessing
    )

    # Process loop with async CPU post-processing
    print(f"\n🚀 Starting extraction ({len(loader)} batches)...")
    print("   GPU/CPU pipelining enabled for maximum throughput")
    print("   Async HuggingFace uploads enabled (⬆ in progress bar = uploading)")
    if config.upload_size_mb:
        print(f"   Size-based upload: every {config.upload_size_mb:.1f} MB")

    async_processor = AsyncPostProcessor(max_workers=2)
    pbar = tqdm(total=len(df), desc="Extracting logits")

    # Track moving average of metrics for display
    avg_model_ms = 0.0
    avg_postprocess_ms = 0.0
    avg_tok_per_sec = 0.0
    ema_alpha = 0.1  # Exponential moving average smoothing
    timed_out = False

    try:
        for batch_idx, (batch, indexes) in enumerate(loader):
            interrupt_handler.check_and_exit()

            # Check for timed shutdown
            if timed_shutdown.check_timeout():
                timed_out = True
                break

            # GPU processing for current batch
            gpu_result, gpu_metrics = gpu_process_batch(
                model,
                batch,
                indexes,
                config,
                exclude_tokens=exclude_tokens,
                think_token_ids=think_tokens_tensor,
            )

            # Update progress bar immediately after GPU returns (overlaps with async CPU work)
            batch_size = batch["input_ids"].size(0)
            num_tokens = gpu_metrics.num_tokens

            # Build postfix dict
            postfix = {
                "batch": f"{batch_size}x{num_tokens}tok",
                "gpu": f"{gpu_metrics.model_time_ms:.0f}ms",
                "cpu": f"{avg_postprocess_ms:.0f}ms",
                "tok/s": f"{avg_tok_per_sec:.0f}",
            }

            # Show buffer size (in MB if size-based, else count) and upload status
            if config.upload_size_mb:
                buf_str = f"{uploader.get_buffer_size_mb():.1f}MB"
            else:
                buf_str = str(len(uploader.buffer))
            if uploader.is_uploading():
                buf_str += "⬆"  # Indicate upload in progress
            postfix["buf"] = buf_str

            # Show remaining time if timed shutdown is enabled
            remaining = timed_shutdown.get_remaining_minutes()
            if remaining is not None:
                postfix["left"] = f"{remaining:.0f}m"

            pbar.update(batch_size)
            pbar.set_postfix(postfix, refresh=True)

            if gpu_result is not None:
                # Submit for async CPU post-processing (returns previous batch results)
                prev_result = async_processor.submit(gpu_result, config, gpu_metrics)

                # Process previous batch results if available
                if prev_result is not None:
                    results, metrics = prev_result
                    for sample in results:
                        uploader.add_sample(sample)

                    if uploader.should_flush():
                        uploader.flush()

                    # Update moving averages from completed batch
                    avg_model_ms = ema_alpha * metrics.model_time_ms + (1 - ema_alpha) * avg_model_ms
                    avg_postprocess_ms = ema_alpha * metrics.postprocess_time_ms + (1 - ema_alpha) * avg_postprocess_ms
                    avg_tok_per_sec = ema_alpha * metrics.model_tokens_per_sec + (1 - ema_alpha) * avg_tok_per_sec

            # Periodic cleanup
            if (batch_idx + 1) % 1000 == 0:
                torch.cuda.empty_cache()

        # Flush remaining async work
        final_result = async_processor.flush()
        if final_result is not None:
            results, _ = final_result
            for sample in results:
                uploader.add_sample(sample)

    finally:
        async_processor.shutdown()
        pbar.close()

    # Handle timed shutdown
    if timed_out:
        timed_shutdown.shutdown_gracefully()
        uploader.shutdown()
        return

    # Final flush (synchronous)
    uploader.flush(final=True)
    uploader.shutdown()

    print(f"\n✅ Extraction complete! Total samples: {uploader.total_samples_uploaded}")
    print(f"📁 Dataset: https://huggingface.co/datasets/{config.hf_output_repo}")


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract logits from a language model for knowledge distillation"
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--dataset", type=str, required=True, help="HuggingFace dataset ID"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="HuggingFace dataset repo for output"
    )

    # Processing options
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.98,
        help="Nucleus sampling threshold (cumulative probability)",
    )
    parser.add_argument(
        "--top-p-max",
        type=int,
        default=100,
        help="Hard maximum number of elements for nucleus sampling",
    )
    parser.add_argument(
        "--sampled-n",
        type=int,
        default=24,
        help="Number of additional logits to sample from remaining vocab",
    )

    # Batch sizing - simple options
    parser.add_argument(
        "--batch-thresholds",
        type=str,
        default=None,
        help='JSON dict of length->samples_per_batch (e.g., \'{"1024":4,"512":8}\')',
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Simple max batch size override (applies to all lengths)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Skip samples longer than this (prevent OOM)",
    )

    # Upload options
    parser.add_argument(
        "--upload-every", type=int, default=500, help="Upload to HF every N samples"
    )
    parser.add_argument(
        "--upload-size-mb",
        type=float,
        default=None,
        help="Upload when buffer reaches this size in MB (overrides --upload-every)",
    )

    # Timed shutdown
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=None,
        help="Stop after this many minutes and save unflushed data locally (useful for time-limited servers). On next --resume, pending data is uploaded first.",
    )

    # Model loading
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to load model on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype (bfloat16, float16, float32)",
    )
    parser.add_argument(
        "--attn", type=str, default="flash_attention_2", help="Attention implementation"
    )

    # Dataset configuration
    parser.add_argument(
        "--user-col", type=str, default="question", help="Column name for user messages"
    )
    parser.add_argument(
        "--assistant-col",
        type=str,
        default="answer",
        help="Column name for assistant messages",
    )
    parser.add_argument(
        "--index-col", type=str, default=None, help="Column name for original index"
    )
    parser.add_argument(
        "--subject-col",
        type=str,
        default=None,
        help="Column name for subject (for sampling)",
    )

    # Pre-computed lengths
    parser.add_argument(
        "--length-col",
        type=str,
        default=None,
        help="Column name for pre-computed token lengths",
    )
    parser.add_argument(
        "--tokenized-dataset",
        type=str,
        default=None,
        help="Dataset ID with pre-computed lengths",
    )

    # Sampling
    parser.add_argument(
        "--sample-subject",
        type=str,
        default=None,
        help='JSON dict of subject->n_samples (e.g., \'{"":20000,"*":6000}\')',
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit total samples to process"
    )
    parser.add_argument(
        "--start-offset", type=int, default=0, help="Skip first N samples"
    )
    parser.add_argument(
        "--full-dataset", action="store_true", help="Process full dataset (no sampling)"
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample instead of taking first N (applies to global or group limits)",
    )

    # Token exclusion
    parser.add_argument(
        "--include-think",
        action="store_true",
        help="Include thinking tags in extraction",
    )
    parser.add_argument(
        "--include-end",
        action="store_true",
        help="Include end-of-generation tokens in extraction",
    )

    # Resume/continue
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous checkpoint"
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing progress and restart",
    )

    # Other
    parser.add_argument(
        "--verify-mask",
        action="store_true",
        help="Verify assistant mask before extraction",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".logit_extraction_cache",
        help="Local cache directory",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Path to custom chat template file (Jinja2 format with {%% generation %%} markers)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse batch thresholds
    batch_thresholds = None
    if args.batch_thresholds:
        batch_thresholds = json.loads(args.batch_thresholds)
    elif args.max_batch_size:
        # Simple override: use the same max batch size for all lengths
        batch_thresholds = {
            8192: 1,
            4096: min(2, args.max_batch_size),
            2048: min(4, args.max_batch_size),
            1024: min(8, args.max_batch_size),
            512: args.max_batch_size,
            256: args.max_batch_size,
        }

    # Parse subject sampling
    sample_per_subject = None
    if args.sample_subject:
        sample_per_subject = json.loads(args.sample_subject)

    config = ExtractionConfig(
        model_id=args.model,
        dataset_id=args.dataset,
        hf_output_repo=args.output,
        dataset_split=args.split,
        top_p=args.top_p,
        top_p_max_elements=args.top_p_max,
        sampled_n=args.sampled_n,
        batch_budget_thresholds=batch_thresholds
        or {8192: 1, 4096: 1, 2048: 2, 1024: 4, 512: 8, 256: 16},
        upload_every_n_samples=args.upload_every,
        device=args.device,
        torch_dtype=args.dtype,
        attn_implementation=args.attn,
        seed=args.seed,
        user_column=args.user_col,
        assistant_column=args.assistant_col,
        index_column=args.index_col,
        subject_column=args.subject_col,
        length_column=args.length_col,
        tokenized_dataset_id=args.tokenized_dataset,
        sample_per_subject=sample_per_subject,
        limit_samples=args.limit if not args.full_dataset else None,
        start_offset=args.start_offset,
        max_seq_len=args.max_seq_len,
        exclude_think_tags=not args.include_think,
        exclude_end_tokens=not args.include_end,
        resume=args.resume,
        force_restart=args.force_restart,
        cache_dir=args.cache_dir,
        chat_template_file=args.chat_template,
        random_sample=args.random_sample,
        # Size-based upload and timed shutdown
        upload_size_mb=args.upload_size_mb,
        max_runtime_minutes=args.max_runtime_minutes,
    )

    run_extraction(config, verify_mask=args.verify_mask)


if __name__ == "__main__":
    main()
