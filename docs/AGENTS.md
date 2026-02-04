# Agent Developer Guide

This document is an entry point for AI coding agents to understand and modify this codebase. Read this first before implementing any features or fixes.

## Repository Purpose

This repository extracts token-level logits (probability distributions) from Large Language Models in a compressed format suitable for knowledge distillation. The key insight is that most tokens have highly concentrated probability mass, so we only need to store:

1. **Nucleus tokens**: The top tokens covering ~98% of probability mass
2. **Sampled tokens**: A random sample from the remaining vocabulary

## File Structure

```
logits/
├── pyproject.toml         # Dependencies and project config (use with uv or pip)
├── extract_logits.py      # Main extraction pipeline (GPU inference + compression)
├── explore_logits.py      # Analysis and visualization tools
├── compute_lengths.py     # Pre-compute token lengths for efficient batching
├── README.md              # User documentation
└── docs/
    └── AGENTS.md          # This file
```

## Core Components

### 1. extract_logits.py (1600 lines)

**Purpose**: Run model inference on a dataset and extract compressed logits.

**Key Classes**:

| Class | Location | Purpose |
|-------|----------|---------|
| `ExtractionConfig` | L128-215 | Dataclass holding all configuration options |
| `StreamingHFUploader` | L222-615 | Handles incremental upload to HuggingFace Hub with async support and local caching |
| `GracefulInterrupt` | L623-651 | Signal handler for graceful Ctrl+C shutdown with local data preservation |
| `TimedShutdown` | L654-706 | Automatic shutdown after configurable runtime with local checkpoint |
| `ConversationDataset` | L713-741 | PyTorch Dataset wrapping conversations |
| `TokenBudgetBatchSampler` | L744-828 | Dynamic batch sizing based on sequence length |
| `GPUBatchResult` | L929-947 | Dataclass for GPU→CPU transfer |
| `AsyncPostProcessor` | L1199-1245 | Async CPU post-processing while GPU runs next batch |

**Key Functions**:

| Function | Location | Purpose |
|----------|----------|---------|
| `pack_indices()` | L86-120 | Compress 32-bit indices to uint16 + packed 2-bit high bits |
| `gpu_process_batch()` | L949-1123 | GPU processing: forward pass, nucleus extraction, sampling |
| `cpu_postprocess_batch()` | L1126-1196 | CPU post-processing: pack indices, create output dicts |
| `prepare_dataset()` | L1273-1406 | Load dataset, apply filters, sort by length |
| `run_extraction()` | L1414-1644 | Main orchestration loop |
| `parse_args()` | L1652-1838 | CLI argument parsing |

**Data Flow**:
```
Dataset → prepare_dataset() → ConversationDataset → TokenBudgetBatchSampler
    → DataLoader → gpu_process_batch() → AsyncPostProcessor
    → StreamingHFUploader → HuggingFace Hub
```

**Async Upload Pipeline**:
The `StreamingHFUploader` uses a background `ThreadPoolExecutor` for non-blocking uploads:
```
Buffer fills → flush() called → previous upload awaited → buffer converted to parquet
    → upload submitted to executor → processing continues immediately
    → final flush() is synchronous to ensure completion
```

**Local Caching & Graceful Shutdown**:
On interruption (Ctrl+C, SIGTERM) or timed shutdown, unflushed data is preserved:
```
Interrupt/Timeout → save_pending_locally() → .logit_extraction_cache/pending/pending-XXXXX.parquet
    → Next run with --resume → load_pending_into_buffer() → data merged with new samples → uploaded together
```

**Critical Implementation Details**:

1. **Assistant Mask**: Uses `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)` to identify which tokens are model outputs. Only these tokens have logits extracted.

2. **Double Precision Computation** (L1014-1017): All probability and logsumexp computations use float64 for numerical accuracy, avoiding issues with peaked distributions where float32 would show top_logit == logsumexp.

3. **Nucleus Extraction** (L1022-1056): Vectorized on GPU in double precision, no Python loops:
   ```python
   sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
   cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
   nucleus_mask = cumsum_probs <= config.top_p
   ```

4. **Sampling from Remaining Vocab** (L1072-1086): Uses Gumbel-max trick in log-space (double precision) to sample from non-nucleus tokens proportionally to their true probabilities, avoiding underflow issues with tiny probabilities.

5. **Quantization**: Logits quantized to uint16 per-row with min/max scaling:
   ```python
   scaled = ((logits - min) / (max - min) * 65535).round()
   ```

6. **Async Upload** (L336-416): Uploads run in a background thread while GPU/CPU processing continues:
   - `_do_upload()`: Performs actual HuggingFace upload, saves backup on failure
   - `_wait_for_pending_upload()`: Blocks until previous upload completes (called before new upload)
   - Buffer is cleared immediately after converting to parquet bytes, not after upload completes

7. **Local Caching on Interruption** (L418-514):
   - `save_pending_locally()`: Saves unflushed buffer to `.logit_extraction_cache/pending/` directory
   - `load_pending_into_buffer()`: On `--resume`, loads pending files back into buffer and deletes them
   - Pending data merges with new samples and uploads together (avoids tiny chunks)

8. **Resume & Dataset Verification Optimizations** (L548-615):
   - Uses `ParquetFragmentScanOptions` with prefetching for faster streaming reads
   - Batch iteration (`ds.batch(batch_size=10000)`) instead of row-by-row (~10x faster)
   - Only fetches `index` column to minimize data transfer
   - Combines local checkpoint with HuggingFace verification for robust resume

9. **Size-Based Upload Threshold** (L277-288):
   - `--upload-size-mb` triggers upload when buffer reaches target size in MB
   - More predictable chunk sizes vs sample-count-based upload
   - `_estimate_sample_size()` tracks buffer size incrementally

### 2. explore_logits.py (1400 lines)

**Purpose**: Load, decompress, and analyze extracted logits.

**Key Functions**:

| Function | Location | Purpose |
|----------|----------|---------|
| `unpack_indices()` | L48-134 | Reverse of `pack_indices()` - reconstruct full token IDs |
| `load_logit_dataset()` | L142-162 | Load HF dataset with streaming support |
| `get_sample()` | L165-266 | Get single sample with automatic unpacking |
| `dequantize_logits()` | L331-350 | Convert uint16 back to float32 |
| `dequantize_top_logits()` | L353-381 | Get nucleus indices+logits for a token |
| `dequantize_sampled_logits()` | L384-411 | Get sampled indices+logits for a token |
| `logits_to_probs()` | L454-477 | Convert logits to probabilities using logsumexp |
| `analyze_nucleus_sizes()` | L1341-1533 | Analyze nucleus size distribution with optional GPU acceleration |
| `analyze_token_probabilities()` | L848-1094 | Align logits with original text |
| `visualize_conversation_probabilities()` | L1097-1269 | Color-coded visualization |

**Key Data Structures** (after `get_sample()` unpacking):

```python
sample = {
    "index": int,           # Original dataset index
    "num_tokens": int,      # Number of extracted tokens
    "top_indices": List[np.ndarray],    # [array([...]), array([...]), ...] per token
    "top_logits_quantized": List[np.ndarray],  # Same structure
    "top_min": List[float], # Min logit per token (for dequantization)
    "top_max": List[float], # Max logit per token
    "sampled_indices": List[np.ndarray],
    "sampled_logits_quantized": List[np.ndarray],
    "sampled_min": List[float],
    "sampled_max": List[float],
    "logsumexp": List[float],  # For probability computation
}
```

### 3. compute_lengths.py (300 lines)

**Purpose**: Pre-compute token lengths for efficient batch ordering.

**Key Functions**:

| Function | Location | Purpose |
|----------|----------|---------|
| `get_transform_fn()` | L99-155 | Creates batch transform for tokenization |
| `main()` | L158-300 | Load dataset, compute lengths, upload |

**Why This Matters**: Pre-computed lengths allow sorting examples by length before extraction. This enables dynamic batch sizing (larger batches for shorter sequences) and minimizes padding waste.

## Common Modification Patterns

### Adding a New CLI Argument

1. Add to `parse_args()` in extract_logits.py (~L1652)
2. Add corresponding field to `ExtractionConfig` dataclass (~L128)
3. Wire it up in `main()` when creating the config (~L1864)

### Key CLI Options for Robustness

**Resumable Extraction**:
```bash
# First run (or interrupted run)
python extract_logits.py --model ... --dataset ... --output ...

# Resume from where it left off
python extract_logits.py --model ... --dataset ... --output ... --resume
```

**Timed Execution** (for time-limited environments):
```bash
# Run for 55 minutes, then save and exit gracefully
python extract_logits.py ... --max-runtime-minutes 55 --resume
```

**Size-Based Uploads** (for consistent chunk sizes):
```bash
# Upload when buffer reaches ~100MB instead of every N samples
python extract_logits.py ... --upload-size-mb 100
```

### Modifying Nucleus Extraction

The nucleus extraction logic is in `gpu_process_batch()` at L762-796. Key variables:
- `config.top_p`: Cumulative probability threshold (default 0.98)
- `config.top_p_max_elements`: Hard cap on nucleus size (default 100)

### Adding a New Output Column

1. Add to the PyArrow schema in `StreamingHFUploader.flush()` at L258-277
2. Compute the value in `cpu_postprocess_batch()` at L866-936
3. Include in the result dict returned from that function

### Supporting Multi-turn Conversations

Currently hardcoded to single-turn. To support multi-turn:
1. Modify `ConversationDataset.__getitem__()` at L463-481
2. Update `prepare_dataset()` column handling at L1013-1146
3. Ensure `analyze_token_probabilities()` in explore_logits.py handles multiple turns

### Adding a New Visualization

Add to explore_logits.py after the existing plot functions (~L500-660). Follow the pattern:
1. Take `sample` dict and token index as input
2. Use `dequantize_*` functions to get data
3. Use matplotlib/seaborn for plotting

## Data Format Reference

### Parquet Schema

```python
schema = pa.schema([
    ("index", pa.int32()),
    ("num_tokens", pa.int32()),
    ("top_indices_low", pa.list_(pa.uint16())),
    ("top_indices_high", pa.binary()),
    ("top_logits_quantized", pa.list_(pa.uint16())),
    ("top_counts", pa.list_(pa.uint8())),
    ("top_min", pa.list_(pa.float32())),
    ("top_max", pa.list_(pa.float32())),
    ("sampled_indices_low", pa.list_(pa.uint16())),
    ("sampled_indices_high", pa.binary()),
    ("sampled_logits_quantized", pa.list_(pa.uint16())),
    ("sampled_min", pa.list_(pa.float32())),
    ("sampled_max", pa.list_(pa.float32())),
    ("logsumexp", pa.list_(pa.float64())),  # double precision for accuracy
])
```

### Index Packing Format

Token IDs are split into:
- **Low 16 bits**: Stored as uint16 array
- **High 2 bits**: Packed 4-per-byte (supports vocab up to 262144)

```python
# Packing (extract_logits.py L84-118)
low_bits = (indices & 0xFFFF).astype(np.uint16)
high_parts = (indices >> 16) & 0x03
packed_high = (h[0]) | (h[1] << 2) | (h[2] << 4) | (h[3] << 6)

# Unpacking (explore_logits.py L48-134)
full_index = low_bits | (high_parts << 16)
```

## Testing Changes

### Manual Testing

```python
# Test extraction on small subset
python extract_logits.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset your/dataset \
    --output your/test-output \
    --limit 10 \
    --verify-mask

# Test analysis
python -c "
from explore_logits import *
ds = load_logit_dataset('your/test-output')
sample = get_sample(ds, 0)
print_sample_info(sample)
"
```

### Key Invariants to Check

1. `sample['num_tokens']` should match `len(sample['top_indices'])`
2. Each `sample['top_indices'][i]` should have length `sample['top_counts'][i]`
3. Probabilities from dequantized logits should sum to ~1 (within nucleus)
4. `sum(logits_to_probs(logits, logsumexp))` ≈ `top_p` for nucleus

## Dependencies

Dependencies are managed via `pyproject.toml`. Use `uv` for fast installation:

```bash
# Core dependencies only
uv sync

# With visualization (matplotlib, seaborn, termcolor)
uv sync --extra viz

# Everything (viz + Jupyter)
uv sync --extra all
```

**Core dependencies** (required):
- `torch`: GPU computation
- `transformers`: Model loading, tokenization, chat templates
- `datasets`: HuggingFace dataset loading
- `huggingface_hub`: Upload to Hub
- `pyarrow`, `pandas`: Parquet I/O
- `numpy`: Numerical operations
- `tqdm`: Progress bars

**Optional `[viz]`** (for analysis/visualization):
- `matplotlib`, `seaborn`: Plotting
- `termcolor`: Colored console output

**Optional `[dev]`** (for notebooks):
- `jupyter`, `ipykernel`: Notebook support

## Performance Considerations

1. **GPU Memory**: Controlled by `--max-seq-len` and batch thresholds. Long sequences run with batch_size=1.

2. **CPU Bottleneck**: `cpu_postprocess_batch()` runs async while GPU processes next batch. If CPU is slower, consider reducing nucleus size or sampled count.

3. **Upload Frequency**: Two modes available:
   - `--upload-every N`: Upload every N samples (default 500)
   - `--upload-size-mb X`: Upload when buffer reaches X MB (more predictable chunk sizes)

4. **Streaming**: Both datasets support streaming mode for datasets that don't fit in RAM.

5. **Async Uploads**: Uploads run in background while processing continues. The progress bar shows `⬆` when an upload is in progress. This significantly improves throughput on slow network connections.

6. **Resume Verification**: On `--resume`, the pipeline verifies existing data using optimized batch iteration with prefetching. This is ~10x faster than row-by-row iteration for large datasets.

7. **Timed Execution**: Use `--max-runtime-minutes` for time-limited environments (e.g., spot instances, shared GPUs). Unflushed data is saved locally and uploaded on next `--resume`.

8. **Analysis GPU Acceleration**: `analyze_nucleus_sizes()` supports a `device` parameter for PyTorch GPU acceleration:
   ```python
   # CPU (default)
   stats = analyze_nucleus_sizes("user/my-logits")

   # GPU acceleration (CUDA)
   stats = analyze_nucleus_sizes("user/my-logits", device="cuda")

   # Apple Silicon (MPS)
   stats = analyze_nucleus_sizes("user/my-logits", device="mps")
   ```
   GPU acceleration batches computations across samples for faster histogram and statistics computation on large datasets.

## Common Issues

### "0 tokens extracted"
The assistant mask is empty. Likely need a custom chat template with `{% generation %}` markers.

### OOM on long sequences
Lower `--max-seq-len` or adjust `--batch-thresholds` to use smaller batches for long sequences.

### Slow extraction
Pre-compute lengths with `compute_lengths.py` to enable sorted batching. This dramatically improves GPU utilization.

### Misaligned analysis
Ensure you use the **same chat template** for analysis as was used for extraction.

### Resume not finding previous progress
1. Check that `--resume` flag is passed
2. Verify `.logit_extraction_cache/checkpoint.json` exists (local checkpoint)
3. Check HuggingFace dataset has data uploaded
4. Use `--force-restart` to ignore existing progress if needed

### Pending files not uploading
If `.logit_extraction_cache/pending/` contains parquet files after a crash:
1. Run with `--resume` - pending files are automatically loaded and merged
2. If merge fails, manually inspect files with `pq.read_table()`

### Upload timeouts on slow connections
- Use `--upload-size-mb` with smaller values (e.g., 50MB) for more frequent, smaller uploads
- The async upload system allows processing to continue during uploads
- Failed uploads are saved to `.logit_extraction_cache/backup_*` for manual retry
