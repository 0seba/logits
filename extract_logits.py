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
import argparse
import io
import json
import os
import signal
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from numpy.random import PCG64
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    batch_budget_thresholds: dict = field(default_factory=lambda: {
        8192: 1,   # Very long sequences: single sample
        4096: 1,   # Long sequences: single sample  
        2048: 2,   # Medium-long: batch of 2
        1024: 4,   # Medium: batch of 4
        512: 8,    # Short: batch of 8
        256: 16,   # Very short: batch of 16
    })

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

        # Ensure cache dir exists
        os.makedirs(config.cache_dir, exist_ok=True)

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
        self.processed_indices.add(sample["index"])

    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        return len(self.buffer) >= self.config.upload_every_n_samples

    def flush(self, final: bool = False):
        """Upload buffered samples as a Parquet file."""
        if not self.buffer:
            return

        # Convert to DataFrame and then Parquet
        df = pd.DataFrame(self.buffer)

        # Convert numpy arrays to lists for parquet compatibility
        # Note: top_* columns now contain variable-length lists due to nucleus sampling
        for col in ["top_indices", "top_logits_quantized", "sampled_indices",
                    "sampled_logits_quantized"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: list(x) if isinstance(x, np.ndarray) else x)

        # Convert to PyArrow
        table = pa.Table.from_pandas(df)

        # Write to bytes
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)

        filename = f"data/part-{self.part_number:05d}.parquet"

        try:
            self.api.upload_file(
                path_or_fileobj=buf,
                path_in_repo=filename,
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            self.total_samples_uploaded += len(self.buffer)
            print(f"\n📤 Uploaded {filename} ({len(self.buffer)} samples, "
                  f"total: {self.total_samples_uploaded})")
        except Exception as e:
            print(f"\n⚠️ Upload failed for {filename}: {e}")
            # Save locally as backup
            backup_path = os.path.join(
                self.config.cache_dir,
                f"backup_{filename.replace('/', '_')}"
            )
            with open(backup_path, 'wb') as f:
                buf.seek(0)
                f.write(buf.read())
            print(f"   Saved backup to {backup_path}")

        self.buffer = []
        self.part_number += 1

        # Save local and remote checkpoint
        self._save_checkpoint()

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
            }
        }

        # Save local checkpoint
        local_path = self._local_checkpoint_path()
        with open(local_path, 'w') as f:
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
                    print(f"📍 Loaded local checkpoint: {len(processed_indices)} samples")
            except Exception as e:
                print(f"Note: Could not load local checkpoint: {e}")

        # Then, verify against HuggingFace dataset
        try:
            ds = load_dataset(self.repo_id, split="train", streaming=True)
            hf_indices = set()
            for row in tqdm(ds, desc="Verifying HF dataset"):
                hf_indices.add(row["index"])
                total_existing += 1

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
            print(f"\n✅ Saved progress. {self.uploader.total_samples_uploaded} samples uploaded.")
            print(f"   Run with --resume to continue from where you left off.")
            sys.exit(0)


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
        # Sort thresholds by length descending (longest first)
        self.thresholds = sorted(thresholds.items(), key=lambda x: -x[0])
        self.drop_last = drop_last
        # Pre-compute number of batches for __len__
        self._num_batches = self._compute_num_batches()

    def _get_max_batch_size(self, length: int) -> int:
        """Get max batch size for a given sequence length."""
        for thr, cnt in self.thresholds:
            if length <= thr:
                return cnt
        return 1  # Default for very long sequences

    def _compute_num_batches(self) -> int:
        """Compute total number of batches."""
        if not self.lengths:
            return 0
        
        num_batches = 0
        batch_size = 0
        current_max = float('inf')
        
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
        current_max = float('inf')
        
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

def verify_assistant_mask(tokenizer, sample_conversation: list[dict], chat_template: Optional[str] = None) -> dict:
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
    print(non_assistant_text[:500] + "..." if len(non_assistant_text) > 500 else non_assistant_text)
    print("=" * 60)

    # Sanity checks
    expected_response = sample_conversation[-1]["content"]
    if expected_response.strip() in assistant_text.strip():
        print("✅ Assistant mask correctly captures the assistant response")
    else:
        print("⚠️ Warning: Assistant response may not be fully captured in assistant tokens")

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
    Process a batch and extract logits.

    Args:
        model: The language model
        tokenizer: The tokenizer
        batch: Tokenized batch with input_ids, attention_mask, assistant_masks
        indexes: Original dataset indices for each sample in batch
        config: Extraction configuration
        exclude_tokens: Token IDs to exclude from extraction (e.g., <|im_end|>)
        think_token_ids: Token IDs for thinking tags to exclude

    Returns:
        List of sample dicts ready for upload, or None if nothing to extract
    """
    device = next(model.parameters()).device

    # CRITICAL: Disable gradient computation to save memory
    with torch.no_grad():
        # Forward pass to get hidden states
        outputs = model.model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state

        # Build extraction mask from assistant_masks
        assistant_mask = batch["assistant_masks"].bool()

        # Exclude end-of-generation tokens if configured
        if config.exclude_end_tokens and exclude_tokens is not None:
            is_not_after_special = ~torch.isin(
                batch["input_ids"].roll(1), exclude_tokens
            )
            assistant_mask = assistant_mask & is_not_after_special

        # Exclude thinking tag positions if configured
        if config.exclude_think_tags and think_token_ids is not None:
            is_not_think = torch.ones_like(batch["input_ids"], dtype=torch.bool)
            for think_id in think_token_ids:
                is_not_think &= ~torch.isin(batch["input_ids"], think_id)
                for shift in range(1, 4):  # Also exclude a few tokens after
                    is_not_think &= ~torch.isin(batch["input_ids"].roll(shift), think_id)
            assistant_mask = assistant_mask & is_not_think

        # Get logits for assistant positions only
        extract_positions = assistant_mask.nonzero(as_tuple=True)
        if extract_positions[0].numel() == 0:
            return None

        logits = model.lm_head(hidden_states[extract_positions])  # (N, vocab)

        # Compute logsumexp for normalization info
        lse = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        probs = torch.exp(logits.float() - lse)

        # Count tokens per sample in batch
        sizes = assistant_mask.sum(dim=-1).tolist()

        # Nucleus (top-p) sampling: extract top logits that accumulate to p probability
        N, vocab_size = probs.size()
        
        # Sort probabilities in descending order
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        
        # Compute cumulative sum of sorted probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find the nucleus: tokens whose cumulative probability <= top_p
        # We use <= to include the token that pushes us over the threshold
        nucleus_mask = cumsum_probs <= config.top_p
        
        # Always include at least the top token (in case top_p is very small)
        nucleus_mask[:, 0] = True
        
        # Apply hard maximum constraint
        position_mask = torch.arange(vocab_size, device=probs.device).expand(N, -1) < config.top_p_max_elements
        nucleus_mask = nucleus_mask & position_mask
        
        # Extract nucleus indices and values (variable length per row)
        nucleus_idx_list = []
        nucleus_vals_list = []
        nucleus_mask_bool = []
        
        for i in range(N):
            row_nucleus_mask = nucleus_mask[i]
            row_indices = sorted_idx[i][row_nucleus_mask]
            row_values = logits.float()[i][row_indices]
            
            nucleus_idx_list.append(row_indices.cpu().numpy().astype(np.int32))
            nucleus_vals_list.append(row_values.cpu().numpy().astype(np.float32))
            nucleus_mask_bool.append(row_nucleus_mask)
        
        # Create a mask for remaining vocabulary sampling
        # Convert list of masks back to tensor
        nucleus_mask_tensor = torch.stack(nucleus_mask_bool)
        
        # Sample additional tokens from remaining vocabulary (outside nucleus)
        probs_rest = probs.clone()
        probs_rest[nucleus_mask_tensor] = 0.0
        rest_sum = probs_rest.sum(dim=-1, keepdim=True)

        # Handle edge case where all probability is in nucleus
        zero_rest = (rest_sum.squeeze(-1) < 1e-10)
        if zero_rest.any():
            # For these rows, sample uniformly from tokens not in nucleus
            uniform_mask = ~nucleus_mask_tensor[zero_rest]
            counts = uniform_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
            probs_rest[zero_rest] = uniform_mask.float() / counts

        # Renormalize remaining probabilities
        rest_sum = probs_rest.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        probs_rest = probs_rest / rest_sum

        # Sample from remaining distribution
        sampled_idx = torch.multinomial(probs_rest, config.sampled_n)
        sampled_vals = logits.float().gather(-1, sampled_idx)

        # Quantize per-token with row-wise min/max
        def quantize_tensor(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Quantize values to uint16 with per-row min/max."""
            row_min = vals.min(dim=-1, keepdim=True).values
            row_max = vals.max(dim=-1, keepdim=True).values
            denom = (row_max - row_min).clamp(min=1e-10)
            scaled = ((vals - row_min) / denom * UINT16_MAX).round()
            # Store as int32 for parquet compatibility
            quantized = scaled.clamp(0, UINT16_MAX).to(torch.int64).to(torch.int32)
            return quantized, row_min.squeeze(-1), row_max.squeeze(-1)
        
        def quantize_numpy(vals: np.ndarray) -> tuple[np.ndarray, float, float]:
            """Quantize numpy array to uint16 with min/max."""
            row_min = float(vals.min())
            row_max = float(vals.max())
            denom = max(row_max - row_min, 1e-10)
            scaled = ((vals - row_min) / denom * UINT16_MAX).round()
            quantized = scaled.clip(0, UINT16_MAX).astype(np.int32)
            return quantized, row_min, row_max

        # Quantize nucleus (variable-length) - already in numpy format
        nucleus_quant_list = []
        nucleus_min_list = []
        nucleus_max_list = []
        for vals in nucleus_vals_list:
            quant, vmin, vmax = quantize_numpy(vals)
            nucleus_quant_list.append(quant)
            nucleus_min_list.append(vmin)
            nucleus_max_list.append(vmax)
        
        # Quantize sampled (fixed-length) - process as tensor
        samp_quant, samp_min, samp_max = quantize_tensor(sampled_vals)
        samp_quant = samp_quant.cpu().numpy()
        samp_idx_np = sampled_idx.cpu().numpy().astype(np.int32)
        samp_min = samp_min.cpu().numpy().astype(np.float32)
        samp_max = samp_max.cpu().numpy().astype(np.float32)

        lse_np = lse.squeeze(-1).cpu().numpy().astype(np.float32)

        # Split by sample and create output dicts
        results = []
        offset = 0
        for i, size in enumerate(sizes):
            if size == 0:
                continue

            end = offset + size
            
            # Extract variable-length nucleus data for this sample
            sample_nucleus_idx = nucleus_idx_list[offset:end]
            sample_nucleus_quant = nucleus_quant_list[offset:end]
            sample_nucleus_min = nucleus_min_list[offset:end]
            sample_nucleus_max = nucleus_max_list[offset:end]
            
            # Extract fixed-length sampled data for this sample
            sample_samp_idx = samp_idx_np[offset:end]
            sample_samp_quant = samp_quant[offset:end]
            sample_samp_min = samp_min[offset:end]
            sample_samp_max = samp_max[offset:end]
            
            results.append({
                "index": int(indexes[i]),
                "num_tokens": size,
                # Nucleus (top-p) logits - variable length per token
                "top_indices": [idx.tolist() for idx in sample_nucleus_idx],
                "top_logits_quantized": [q.tolist() for q in sample_nucleus_quant],
                "top_min": sample_nucleus_min,
                "top_max": sample_nucleus_max,
                # Sampled logits - fixed length
                "sampled_indices": sample_samp_idx.tolist(),
                "sampled_logits_quantized": sample_samp_quant.tolist(),
                "sampled_min": sample_samp_min.tolist(),
                "sampled_max": sample_samp_max.tolist(),
                "logsumexp": lse_np[offset:end].tolist(),
            })
            offset = end

    return results


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_dataset(config: ExtractionConfig,
                    skip_indices: set[int]) -> tuple[pd.DataFrame, list[int]]:
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
        length_ds = load_dataset(config.tokenized_dataset_id, split=config.dataset_split)
        df["length"] = length_ds[config.length_column]
    elif config.length_column and config.length_column in df.columns:
        df["length"] = df[config.length_column]
    else:
        # Mark for length computation during iteration
        df["length"] = 0

    # Apply stratified sampling if configured
    if config.sample_per_subject and config.subject_column:
        if config.subject_column in df.columns:
            print(f"🎲 Applying sampling by {config.subject_column}")
            random_state = np.random.Generator(PCG64(seed=config.seed))

            def sample_group(group):
                subject = group.name
                n_samples = config.sample_per_subject.get(
                    subject,
                    config.sample_per_subject.get("*", len(group))
                )
                actual = min(n_samples, len(group))
                return group.sample(n=actual, random_state=random_state)

            df = df.groupby(config.subject_column, group_keys=False).apply(
                sample_group, include_groups=True
            )
        else:
            print(f"⚠️ Subject column '{config.subject_column}' not found, skipping sampling")

    # Filter out already processed indices
    if skip_indices:
        original_len = len(df)
        df = df[~df["original_index"].isin(skip_indices)]
        print(f"⏭️  Skipping {original_len - len(df)} already processed samples")

    # Apply start offset
    if config.start_offset > 0:
        df = df.iloc[config.start_offset:]
        print(f"⏭️  Skipped first {config.start_offset} samples")

    # Apply limit if set
    if config.limit_samples:
        df = df.head(config.limit_samples)
        print(f"🔢 Limiting to {len(df)} samples")

    # Filter out samples that are too long
    if config.max_seq_len and "length" in df.columns:
        original_len = len(df)
        df = df[df["length"] <= config.max_seq_len]
        skipped = original_len - len(df)
        if skipped > 0:
            print(f"⏭️  Skipped {skipped} samples longer than {config.max_seq_len} tokens")

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
    print(f"Top-P: {config.top_p}, Max Elements: {config.top_p_max_elements}, Sampled-N: {config.sampled_n}")
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

    # Load model and tokenizer
    print(f"\n🤖 Loading model: {config.model_id}")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
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
        with open(config.chat_template_file, 'r') as f:
            custom_chat_template = f.read()
        print("   ✅ Custom chat template loaded")

    # Get special tokens to exclude
    end_token_strs = ["<|im_end|>", "<|endoftext|>"]
    exclude_token_ids = []
    for s in end_token_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            exclude_token_ids.extend(ids)
    exclude_tokens = torch.tensor(exclude_token_ids, dtype=torch.long) if exclude_token_ids else None

    # Get think token IDs for exclusion
    think_token_ids = []
    for tag in ["<|im_start|>", "<|im_end|>", "</think>", "<|endoftext|>"]:
        ids = tokenizer.encode(tag, add_special_tokens=False)
        if ids:
            think_token_ids.extend(ids)
    think_tokens_tensor = torch.tensor(think_token_ids, dtype=torch.long) if think_token_ids else None

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
        verify_assistant_mask(tokenizer, sample_messages, chat_template=custom_chat_template)
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

    # Process loop
    print(f"\n🚀 Starting extraction ({len(loader)} batches)...")
    pbar = tqdm(loader, desc="Extracting logits")

    for batch_idx, (batch, indexes) in enumerate(loader):
        interrupt_handler.check_and_exit()

        results = process_batch(
            model, tokenizer, batch, indexes, config,
            exclude_tokens=exclude_tokens,
            think_token_ids=think_tokens_tensor,
        )

        if results:
            for sample in results:
                uploader.add_sample(sample)

            if uploader.should_flush():
                uploader.flush()

        pbar.update(batch["input_ids"].size(0))
        pbar.set_postfix({
            "buffer": len(uploader.buffer),
            "uploaded": uploader.total_samples_uploaded
        })

        # Periodic cleanup
        if (batch_idx + 1) % 1000 == 0:
            torch.cuda.empty_cache()

    # Final flush
    uploader.flush(final=True)

    print(f"\n✅ Extraction complete! Total samples: {uploader.total_samples_uploaded}")
    print(f"📁 Dataset: https://huggingface.co/datasets/{config.hf_output_repo}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract logits from a language model for knowledge distillation"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset ID")
    parser.add_argument("--output", type=str, required=True,
                        help="HuggingFace dataset repo for output")

    # Processing options
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--top-p", type=float, default=0.98,
                        help="Nucleus sampling threshold (cumulative probability)")
    parser.add_argument("--top-p-max", type=int, default=100,
                        help="Hard maximum number of elements for nucleus sampling")
    parser.add_argument("--sampled-n", type=int, default=24,
                        help="Number of additional logits to sample from remaining vocab")

    # Batch sizing - simple options
    parser.add_argument("--batch-thresholds", type=str, default=None,
                        help="JSON dict of length->samples_per_batch (e.g., '{\"1024\":4,\"512\":8}')")
    parser.add_argument("--max-batch-size", type=int, default=None,
                        help="Simple max batch size override (applies to all lengths)")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Skip samples longer than this (prevent OOM)")

    # Upload options
    parser.add_argument("--upload-every", type=int, default=500,
                        help="Upload to HF every N samples")

    # Model loading
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load model on")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype (bfloat16, float16, float32)")
    parser.add_argument("--attn", type=str, default="flash_attention_2",
                        help="Attention implementation")

    # Dataset configuration
    parser.add_argument("--user-col", type=str, default="question",
                        help="Column name for user messages")
    parser.add_argument("--assistant-col", type=str, default="answer",
                        help="Column name for assistant messages")
    parser.add_argument("--index-col", type=str, default=None,
                        help="Column name for original index")
    parser.add_argument("--subject-col", type=str, default=None,
                        help="Column name for subject (for sampling)")

    # Pre-computed lengths
    parser.add_argument("--length-col", type=str, default=None,
                        help="Column name for pre-computed token lengths")
    parser.add_argument("--tokenized-dataset", type=str, default=None,
                        help="Dataset ID with pre-computed lengths")

    # Sampling
    parser.add_argument("--sample-subject", type=str, default=None,
                        help="JSON dict of subject->n_samples (e.g., '{\"\":20000,\"*\":6000}')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total samples to process")
    parser.add_argument("--start-offset", type=int, default=0,
                        help="Skip first N samples")
    parser.add_argument("--full-dataset", action="store_true",
                        help="Process full dataset (no sampling)")

    # Token exclusion
    parser.add_argument("--include-think", action="store_true",
                        help="Include thinking tags in extraction")
    parser.add_argument("--include-end", action="store_true",
                        help="Include end-of-generation tokens in extraction")

    # Resume/continue
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous checkpoint")
    parser.add_argument("--force-restart", action="store_true",
                        help="Ignore existing progress and restart")

    # Other
    parser.add_argument("--verify-mask", action="store_true",
                        help="Verify assistant mask before extraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cache-dir", type=str, default=".logit_extraction_cache",
                        help="Local cache directory")
    parser.add_argument("--chat-template", type=str, default=None,
                        help="Path to custom chat template file (Jinja2 format with {% generation %} markers)")

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
        batch_budget_thresholds=batch_thresholds or {
            8192: 1, 4096: 1, 2048: 2, 1024: 4, 512: 8, 256: 16
        },
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
    )

    run_extraction(config, verify_mask=args.verify_mask)


if __name__ == "__main__":
    main()
