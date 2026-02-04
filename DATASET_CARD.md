# Dataset Card

This dataset contains compressed token-level logits extracted from **[MODEL_NAME]** using nucleus sampling. For each assistant token, it stores:

- **Nucleus logits**: Top tokens covering ~98% probability mass (variable size, up to 100)
- **Sampled logits**: 24 randomly sampled tokens from the remaining vocabulary
- **LogSumExp**: For accurate probability reconstruction

## Token Alignment

**Important**: Logits are only extracted for **assistant response tokens**. The token indices in this dataset are shifted relative to the full tokenized conversation.

For example, given this conversation:
```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
The answer is 4.<|im_end|>
```

The full tokenized sequence might be:
```
Index:  0           1      2    3   4    5            6           7     8      9   10  11
Token:  <|im_start|> user   \n   What is  2+2?        <|im_end|>  <|im_start|> assistant \n  The answer ...
        |<------- user prompt (no logits) ------->|              |<-- assistant (logits extracted) -->|
```

In the extracted dataset:
- `token_idx=0` corresponds to "The" (first assistant token)
- `token_idx=1` corresponds to " answer"
- `num_tokens` = total assistant tokens only

The logits at position `i` represent the model's prediction **for** token `i` (i.e., the distribution the model used to generate that token).

## Quick Start (Standalone)

```python
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Load streaming dataset and tokenizer
ds = load_dataset("YOUR_DATASET_ID", split="train", streaming=True)
tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_ID")
sample = next(iter(ds))

# Unpack indices (low 16 bits + packed 2-bit high bits)
def unpack_indices(low, high_bytes, counts):
    packed = np.frombuffer(high_bytes, dtype=np.uint8)
    high = np.stack([(packed >> i) & 0x03 for i in [0, 2, 4, 6]], axis=1).flatten()
    full = np.asarray(low, dtype=np.int32) | (high[:len(low)].astype(np.int32) << 16)
    result, offset = [], 0
    for c in counts:
        result.append(full[offset:offset+c])
        offset += c
    return result

# Dequantize logits
def dequantize(quantized, vmin, vmax):
    return (np.array(quantized, dtype=np.float32) / 65535) * (vmax - vmin) + vmin

# Hydrate sample: unpack all indices and slice logits
def hydrate_sample(sample):
    counts = sample["top_counts"]
    num_tokens = sample["num_tokens"]

    # Unpack nucleus (top-p) indices
    sample["top_indices"] = unpack_indices(
        sample["top_indices_low"], sample["top_indices_high"], counts
    )

    # Slice nucleus logits by counts
    flat_top = np.array(sample["top_logits_quantized"], dtype=np.uint16)
    sample["top_logits"] = []
    offset = 0
    for c in counts:
        sample["top_logits"].append(flat_top[offset:offset+c])
        offset += c

    # Unpack sampled indices (fixed stride per token)
    stride = len(sample["sampled_indices_low"]) // num_tokens
    sampled_counts = [stride] * num_tokens
    sample["sampled_indices"] = unpack_indices(
        sample["sampled_indices_low"], sample["sampled_indices_high"], sampled_counts
    )

    # Slice sampled logits
    flat_sampled = np.array(sample["sampled_logits_quantized"], dtype=np.uint16)
    sample["sampled_logits"] = [flat_sampled[i*stride:(i+1)*stride] for i in range(num_tokens)]

    return sample

# Print nucleus (top-p) logits for a token
def print_top_logits(sample, token_idx, tokenizer):
    indices = sample["top_indices"][token_idx]
    logits = dequantize(sample["top_logits"][token_idx], sample["top_min"][token_idx], sample["top_max"][token_idx])
    lse = sample["logsumexp"][token_idx]
    probs = np.exp(logits - lse) * 100  # Convert to percentage

    # Sort by descending logit
    order = np.argsort(logits)[::-1]

    print(f"=== Nucleus logits for token {token_idx} (logsumexp={lse:.4f}) ===")
    print(f"{'Rank':<6} {'Index':<8} {'Token':<20} {'Logit':<12} {'Prob %':<10}")
    print("-" * 60)
    for rank, i in enumerate(order):
        token_str = repr(tokenizer.decode([indices[i]]))[1:-1]  # Remove quotes
        if len(token_str) > 18:
            token_str = token_str[:15] + "..."
        print(f"{rank:<6} {indices[i]:<8} {token_str:<20} {logits[i]:<12.4f} {probs[i]:<10.4f}")
    print(f"\nTotal nucleus probability mass: {probs.sum():.2f}%")

# Print sampled logits for a token
def print_sampled_logits(sample, token_idx, tokenizer):
    indices = sample["sampled_indices"][token_idx]
    logits = dequantize(sample["sampled_logits"][token_idx], sample["sampled_min"][token_idx], sample["sampled_max"][token_idx])
    lse = sample["logsumexp"][token_idx]
    probs = np.exp(logits - lse) * 100  # Convert to percentage

    # Sort by descending logit
    order = np.argsort(logits)[::-1]

    print(f"=== Sampled logits for token {token_idx} (logsumexp={lse:.4f}) ===")
    print(f"{'#':<6} {'Index':<8} {'Token':<20} {'Logit':<12} {'Prob %':<10}")
    print("-" * 60)
    for rank, i in enumerate(order):
        token_str = repr(tokenizer.decode([indices[i]]))[1:-1]
        if len(token_str) > 18:
            token_str = token_str[:15] + "..."
        print(f"{rank:<6} {indices[i]:<8} {token_str:<20} {logits[i]:<12.4f} {probs[i]:<10.6f}")
    print(f"\nTotal sampled probability mass: {probs.sum():.6f}%")

# Example usage
sample = hydrate_sample(sample)
print(f"Sample has {sample['num_tokens']} tokens\n")

print_top_logits(sample, token_idx=0, tokenizer=tokenizer)
print()
print_sampled_logits(sample, token_idx=0, tokenizer=tokenizer)
```

> **Note**: The `logsumexp` values are computed over the **full vocabulary** during extraction. This allows accurate probability reconstruction: `prob = exp(logit - logsumexp)`. Without it, probabilities from partial logits wouldn't sum correctly.

## Using explore_logits.py

```python
from transformers import AutoTokenizer
from explore_logits import (
    load_logit_dataset, get_sample, print_sample_info,
    dequantize_top_logits, dequantize_sampled_logits, logits_to_probs
)
import numpy as np

# Load streaming dataset and tokenizer
ds = load_logit_dataset("YOUR_DATASET_ID", streaming=True)
tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_ID")
sample = get_sample(ds, index=0)  # Automatically unpacks indices and slices logits

# Print sample overview
print_sample_info(sample)

# Print nucleus (top-p) logits for a token
def print_top_logits(sample, token_idx, tokenizer):
    indices, logits = dequantize_top_logits(sample, token_idx)
    lse = sample["logsumexp"][token_idx]
    probs = logits_to_probs(logits, lse) * 100  # Convert to percentage

    # Sort by descending logit
    order = np.argsort(logits)[::-1]

    print(f"=== Nucleus logits for token {token_idx} (logsumexp={lse:.4f}) ===")
    print(f"{'Rank':<6} {'Index':<8} {'Token':<20} {'Logit':<12} {'Prob %':<10}")
    print("-" * 60)
    for rank, i in enumerate(order):
        token_str = repr(tokenizer.decode([indices[i]]))[1:-1]
        if len(token_str) > 18:
            token_str = token_str[:15] + "..."
        print(f"{rank:<6} {indices[i]:<8} {token_str:<20} {logits[i]:<12.4f} {probs[i]:<10.4f}")
    print(f"\nTotal nucleus probability mass: {probs.sum():.2f}%")

# Print sampled logits for a token
def print_sampled_logits(sample, token_idx, tokenizer):
    indices, logits = dequantize_sampled_logits(sample, token_idx)
    lse = sample["logsumexp"][token_idx]
    probs = logits_to_probs(logits, lse) * 100  # Convert to percentage

    # Sort by descending logit
    order = np.argsort(logits)[::-1]

    print(f"=== Sampled logits for token {token_idx} (logsumexp={lse:.4f}) ===")
    print(f"{'#':<6} {'Index':<8} {'Token':<20} {'Logit':<12} {'Prob %':<10}")
    print("-" * 60)
    for rank, i in enumerate(order):
        token_str = repr(tokenizer.decode([indices[i]]))[1:-1]
        if len(token_str) > 18:
            token_str = token_str[:15] + "..."
        print(f"{rank:<6} {indices[i]:<8} {token_str:<20} {logits[i]:<12.4f} {probs[i]:<10.6f}")
    print(f"\nTotal sampled probability mass: {probs.sum():.6f}%")

# Example usage
print(f"Sample has {sample['num_tokens']} tokens\n")

print_top_logits(sample, token_idx=0, tokenizer=tokenizer)
print()
print_sampled_logits(sample, token_idx=0, tokenizer=tokenizer)
```
