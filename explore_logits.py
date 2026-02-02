"""
Interactive exploration tools for extracted logit data.

Usage in Jupyter notebook:
    from explore_logits import *

    # Load dataset
    ds = load_logit_dataset("seba/dataset-name")

    # View a sample
    sample = get_sample(ds, index=0)
    print_sample_info(sample)

    # Dequantize and inspect logits
    top_logits = dequantize_top_logits(sample, token_idx=0)
    sampled_logits = dequantize_sampled_logits(sample, token_idx=0)

    # Visualize
    plot_logit_distribution(sample, token_idx=0)
    plot_nucleus_vs_sampled(sample, token_idx=0)

    # Get statistics
    stats = compute_dataset_stats(ds)
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from typing import Optional, Union, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Constants
UINT16_MAX = 65535

# =============================================================================
# Helper: Index Unpacking (for new packed format)
# =============================================================================


def unpack_indices(low_bits_list, high_bits_list, sizes_list=None):
    """
    Unpack indices from low (uint16) and high (packed 2-bit) components.

    Args:
        low_bits_list: List of List/Array of uint16 low bits
        high_bits_list: List of bytes/binary high bits (packed)
        sizes_list: Optional list of expected sizes (if not provided, inferred from low_bits)


    Returns:
        List of numpy arrays (int32) containing the full indices
    """
    # Check if inputs are flattened (single array/bytes) vs list of arrays
    is_flattened = isinstance(low_bits_list, (np.ndarray, list)) and (
        len(low_bits_list) > 0 and not isinstance(low_bits_list[0], (np.ndarray, list))
    )

    # If flattened, we need sizes to slice
    if is_flattened:
        if sizes_list is None:
            raise ValueError("sizes_list required for unpacking flattened indices")

        # Unpack high bits from single binary blob
        if isinstance(high_bits_list, (bytes, bytearray)):
            packed_arr = np.frombuffer(high_bits_list, dtype=np.uint8)
        else:
            packed_arr = np.array(high_bits_list, dtype=np.uint8)

        # Unpack all high bits at once
        hi0 = packed_arr & 0x03
        hi1 = (packed_arr >> 2) & 0x03
        hi2 = (packed_arr >> 4) & 0x03
        hi3 = (packed_arr >> 6) & 0x03

        all_high_parts = np.stack([hi0, hi1, hi2, hi3], axis=1).flatten()
        all_low_bits = np.asarray(low_bits_list, dtype=np.int32)

        # Trim high parts to match low bits length
        all_high_parts = all_high_parts[: len(all_low_bits)]

        # Combine
        all_full_indices = all_low_bits | (all_high_parts.astype(np.int32) << 16)

        # Slice into per-token arrays
        unpacked_indices = []
        offset = 0
        for size in sizes_list:
            end = offset + size
            unpacked_indices.append(all_full_indices[offset:end])
            offset = end

        return unpacked_indices

    # Legacy (List of Lists) handling
    unpacked_indices = []

    for i, (low, high_bytes) in enumerate(zip(low_bits_list, high_bits_list)):
        # Ensure low is numpy array
        low = np.asarray(low, dtype=np.int32)

        # Unpack high bits
        if isinstance(high_bytes, (bytes, bytearray)):
            packed_arr = np.frombuffer(high_bytes, dtype=np.uint8)
        else:
            # Handle case where it might be already converted or different format
            packed_arr = np.array(high_bytes, dtype=np.uint8)

        # Unpack 4 items per byte
        # (d << 6) | (c << 4) | (b << 2) | a
        hi0 = packed_arr & 0x03
        hi1 = (packed_arr >> 2) & 0x03
        hi2 = (packed_arr >> 4) & 0x03
        hi3 = (packed_arr >> 6) & 0x03

        # Interleave to get original sequence
        high_parts = np.stack([hi0, hi1, hi2, hi3], axis=1).flatten()

        # Trim to actual length
        target_len = len(low)
        high_parts = high_parts[:target_len]

        # Combine
        full_idx = low | (high_parts.astype(np.int32) << 16)
        unpacked_indices.append(full_idx)

    return unpacked_indices


# =============================================================================
# Loading and Basic Access
# =============================================================================


def load_logit_dataset(dataset_id: str, split: str = "train", streaming: bool = False):
    """
    Load a logit extraction dataset from HuggingFace Hub.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "seba/dataset-name")
        split: Dataset split to load (default: "train")
        streaming: Whether to use streaming mode (default: False)

    Returns:
        Dataset object

    Example:
        ds = load_logit_dataset("seba/my-logits")
        print(f"Loaded {len(ds)} samples")
    """
    print(f"📦 Loading dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split=split, streaming=streaming)
    if not streaming:
        print(f"✅ Loaded {len(ds)} samples")
    return ds


def get_sample(dataset, index: int = 0) -> Dict[str, Any]:
    """
    Get a single sample from the dataset.

    Args:
        dataset: The dataset object
        index: Sample index to retrieve

    Returns:
        Dictionary with sample data

    Example:
        sample = get_sample(ds, index=42)
    """
    if hasattr(dataset, "__getitem__"):
        item = dataset[index]
    else:
        # Streaming dataset - iterate to index
        item = None
        for i, curr in enumerate(dataset):
            if i == index:
                item = curr
                break
        if item is None:
            raise IndexError(f"Index {index} out of range")

    # Hydrate packed/flattened indices if present
    if "top_indices_low" in item and "top_indices" not in item:
        if "top_counts" in item:
            # Flattened format with counts
            counts = item["top_counts"]
            item["top_indices"] = unpack_indices(
                item["top_indices_low"], item["top_indices_high"], sizes_list=counts
            )

            # Also slice logits if they are flattened
            if not isinstance(item["top_logits_quantized"][0], (list, np.ndarray)):
                flat_logits = np.array(item["top_logits_quantized"], dtype=np.uint16)
                item["top_logits_quantized"] = []
                offset = 0
                for size in counts:
                    end = offset + size
                    item["top_logits_quantized"].append(flat_logits[offset:end])
                    offset = end
        else:
            # Legacy list-of-lists packed format
            item["top_indices"] = unpack_indices(
                item["top_indices_low"], item["top_indices_high"]
            )

    if "sampled_indices_low" in item and "sampled_indices" not in item:
        # Determine stride/counts
        total_len = len(item["sampled_indices_low"])
        num_tokens = item["num_tokens"]

        if num_tokens > 0:
            stride = total_len // num_tokens
            counts = [stride] * num_tokens

            item["sampled_indices"] = unpack_indices(
                item["sampled_indices_low"],
                item["sampled_indices_high"],
                sizes_list=counts,
            )

            # Slice logits if flattened
            if not isinstance(item["sampled_logits_quantized"][0], (list, np.ndarray)):
                flat_logits = np.array(
                    item["sampled_logits_quantized"], dtype=np.uint16
                )
                item["sampled_logits_quantized"] = []
                offset = 0
                for size in counts:
                    end = offset + size
                    item["sampled_logits_quantized"].append(flat_logits[offset:end])
                    offset = end
        else:
            item["sampled_indices"] = []
            item["sampled_logits_quantized"] = []

    return item


def print_sample_info(sample: Dict[str, Any], verbose: bool = True):
    """
    Print information about a sample.

    Args:
        sample: Sample dictionary
        verbose: Whether to print detailed info (default: True)

    Example:
        sample = get_sample(ds, 0)
        print_sample_info(sample)
    """
    print("=" * 60)
    print(f"SAMPLE INFO - Index: {sample['index']}")
    print("=" * 60)
    print(f"Number of tokens: {sample['num_tokens']}")

    # Top-p (nucleus) logits
    top_indices = sample["top_indices"]
    if isinstance(top_indices[0], list):
        nucleus_sizes = [len(indices) for indices in top_indices]
        print(f"\nNucleus (top-p) logits:")
        print(f"  - Variable length per token")
        print(
            f"  - Min size: {min(nucleus_sizes)}, Max size: {max(nucleus_sizes)}, Avg: {np.mean(nucleus_sizes):.1f}"
        )
    else:
        print(f"\nNucleus (top-p) logits: {len(top_indices)} per token")

    # Sampled logits
    sampled_indices = sample["sampled_indices"]
    if isinstance(sampled_indices[0], list):
        print(f"Sampled logits: {len(sampled_indices[0])} per token")
    else:
        print(f"Sampled logits: {len(sampled_indices)} total")

    if verbose:
        print("\nData structure:")
        print(f"  - top_indices: {type(top_indices)} of length {len(top_indices)}")
        print(f"  - top_logits_quantized: {type(sample['top_logits_quantized'])}")
        print(
            f"  - top_min: {type(sample['top_min'])}, length {len(sample['top_min'])}"
        )
        print(
            f"  - top_max: {type(sample['top_max'])}, length {len(sample['top_max'])}"
        )
        print(f"  - sampled_indices: {type(sampled_indices)}")
        print(
            f"  - sampled_logits_quantized: {type(sample['sampled_logits_quantized'])}"
        )
        print(
            f"  - logsumexp: {type(sample['logsumexp'])}, length {len(sample['logsumexp'])}"
        )

    print("=" * 60)


# =============================================================================
# Dequantization
# =============================================================================


def dequantize_logits(quantized: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Dequantize uint16 logits back to float32.

    Args:
        quantized: Quantized logits (uint16/int32 format)
        vmin: Minimum value used during quantization
        vmax: Maximum value used during quantization

    Returns:
        Dequantized logits as float32 array

    Example:
        logits = dequantize_logits(sample['top_logits_quantized'][0],
                                   sample['top_min'][0],
                                   sample['top_max'][0])
    """
    quantized = np.array(quantized, dtype=np.float32)
    denom = max(vmax - vmin, 1e-10)
    return (quantized / UINT16_MAX) * denom + vmin


def dequantize_top_logits(
    sample: Dict[str, Any], token_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dequantize nucleus (top-p) logits for a specific token.

    Args:
        sample: Sample dictionary
        token_idx: Token index (0 to num_tokens-1)

    Returns:
        Tuple of (indices, logits) as numpy arrays

    Example:
        indices, logits = dequantize_top_logits(sample, token_idx=0)
        print(f"Top token: {indices[0]} with logit: {logits[0]:.3f}")
    """
    if token_idx >= sample["num_tokens"]:
        raise IndexError(
            f"Token index {token_idx} out of range (num_tokens={sample['num_tokens']})"
        )

    indices = np.array(sample["top_indices"][token_idx], dtype=np.int32)
    quantized = np.array(sample["top_logits_quantized"][token_idx], dtype=np.int32)
    vmin = sample["top_min"][token_idx]
    vmax = sample["top_max"][token_idx]

    logits = dequantize_logits(quantized, vmin, vmax)
    return indices, logits


def dequantize_sampled_logits(
    sample: Dict[str, Any], token_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dequantize sampled logits for a specific token.

    Args:
        sample: Sample dictionary
        token_idx: Token index (0 to num_tokens-1)

    Returns:
        Tuple of (indices, logits) as numpy arrays

    Example:
        indices, logits = dequantize_sampled_logits(sample, token_idx=0)
    """
    if token_idx >= sample["num_tokens"]:
        raise IndexError(
            f"Token index {token_idx} out of range (num_tokens={sample['num_tokens']})"
        )

    indices = np.array(sample["sampled_indices"][token_idx], dtype=np.int32)
    quantized = np.array(sample["sampled_logits_quantized"][token_idx], dtype=np.int32)
    vmin = sample["sampled_min"][token_idx]
    vmax = sample["sampled_max"][token_idx]

    logits = dequantize_logits(quantized, vmin, vmax)
    return indices, logits


def dequantize_all_logits(
    sample: Dict[str, Any], token_idx: int
) -> Dict[str, np.ndarray]:
    """
    Dequantize all logits (nucleus + sampled) for a specific token.

    Args:
        sample: Sample dictionary
        token_idx: Token index (0 to num_tokens-1)

    Returns:
        Dictionary with:
            - 'nucleus_indices': nucleus token indices
            - 'nucleus_logits': nucleus logits
            - 'sampled_indices': sampled token indices
            - 'sampled_logits': sampled logits
            - 'logsumexp': log-sum-exp for normalization

    Example:
        logits = dequantize_all_logits(sample, token_idx=0)
        print(f"Nucleus size: {len(logits['nucleus_indices'])}")
        print(f"Sampled size: {len(logits['sampled_indices'])}")
    """
    nucleus_idx, nucleus_logits = dequantize_top_logits(sample, token_idx)
    sampled_idx, sampled_logits = dequantize_sampled_logits(sample, token_idx)

    return {
        "nucleus_indices": nucleus_idx,
        "nucleus_logits": nucleus_logits,
        "sampled_indices": sampled_idx,
        "sampled_logits": sampled_logits,
        "logsumexp": sample["logsumexp"][token_idx],
    }


# =============================================================================
# Probability Computation
# =============================================================================


def logits_to_probs(
    logits: np.ndarray, logsumexp: Optional[float] = None
) -> np.ndarray:
    """
    Convert logits to probabilities.

    Args:
        logits: Array of logits
        logsumexp: Optional precomputed logsumexp for normalization
                  If None, will compute from logits (may be inaccurate for partial vocab)

    Returns:
        Array of probabilities

    Example:
        indices, logits = dequantize_top_logits(sample, 0)
        probs = logits_to_probs(logits, sample['logsumexp'][0])
    """
    if logsumexp is not None:
        # Use precomputed logsumexp for accurate normalization
        return np.exp(logits - logsumexp)
    else:
        # Compute from available logits (may be inaccurate for partial vocab)
        return np.exp(logits - np.log(np.sum(np.exp(logits))))


def get_nucleus_probability_mass(sample: Dict[str, Any], token_idx: int) -> float:
    """
    Compute the total probability mass captured by the nucleus (top-p) logits.

    Args:
        sample: Sample dictionary
        token_idx: Token index

    Returns:
        Total probability mass in nucleus (should be close to top_p threshold, e.g., 0.98)

    Example:
        mass = get_nucleus_probability_mass(sample, token_idx=0)
        print(f"Nucleus captures {mass*100:.2f}% of probability mass")
    """
    indices, logits = dequantize_top_logits(sample, token_idx)
    lse = sample["logsumexp"][token_idx]
    probs = logits_to_probs(logits, lse)
    return np.sum(probs)


# =============================================================================
# Visualization
# =============================================================================


def plot_logit_distribution(
    sample: Dict[str, Any],
    token_idx: int,
    show_top_n: int = 20,
    figsize: tuple = (12, 5),
):
    """
    Plot the logit distribution for a specific token.

    Args:
        sample: Sample dictionary
        token_idx: Token index
        show_top_n: Number of top tokens to show in detail
        figsize: Figure size (width, height)

    Example:
        plot_logit_distribution(sample, token_idx=0, show_top_n=30)
    """
    logits_dict = dequantize_all_logits(sample, token_idx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Top N nucleus logits
    nucleus_indices = logits_dict["nucleus_indices"][:show_top_n]
    nucleus_logits = logits_dict["nucleus_logits"][:show_top_n]
    nucleus_probs = logits_to_probs(nucleus_logits, logits_dict["logsumexp"])

    ax1.bar(range(len(nucleus_logits)), nucleus_probs, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Token Rank")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"Top {show_top_n} Nucleus Tokens (Token {token_idx})")
    ax1.grid(axis="y", alpha=0.3)

    # Right plot: Sampled logits histogram
    sampled_logits = logits_dict["sampled_logits"]
    sampled_probs = logits_to_probs(sampled_logits, logits_dict["logsumexp"])

    ax2.hist(sampled_probs, bins=30, color="coral", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Probability")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Sampled Logits Distribution (Token {token_idx})")
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary stats
    print(f"\nToken {token_idx} Statistics:")
    print(f"  Nucleus size: {len(logits_dict['nucleus_indices'])}")
    print(
        f"  Nucleus probability mass: {get_nucleus_probability_mass(sample, token_idx):.4f}"
    )
    print(f"  Top token probability: {nucleus_probs[0]:.4f}")
    print(f"  Sampled tokens: {len(logits_dict['sampled_indices'])}")
    print(
        f"  Sampled prob range: [{sampled_probs.min():.2e}, {sampled_probs.max():.2e}]"
    )


def plot_nucleus_vs_sampled(
    sample: Dict[str, Any], token_idx: int, figsize: tuple = (10, 6)
):
    """
    Compare nucleus and sampled logit distributions.

    Args:
        sample: Sample dictionary
        token_idx: Token index
        figsize: Figure size

    Example:
        plot_nucleus_vs_sampled(sample, token_idx=0)
    """
    logits_dict = dequantize_all_logits(sample, token_idx)

    nucleus_probs = logits_to_probs(
        logits_dict["nucleus_logits"], logits_dict["logsumexp"]
    )
    sampled_probs = logits_to_probs(
        logits_dict["sampled_logits"], logits_dict["logsumexp"]
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Plot both distributions
    ax.hist(
        np.log10(nucleus_probs + 1e-10),
        bins=50,
        alpha=0.6,
        label="Nucleus",
        color="steelblue",
    )
    ax.hist(
        np.log10(sampled_probs + 1e-10),
        bins=50,
        alpha=0.6,
        label="Sampled",
        color="coral",
    )

    ax.set_xlabel("Log10(Probability)")
    ax.set_ylabel("Count")
    ax.set_title(f"Nucleus vs Sampled Probability Distributions (Token {token_idx})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_nucleus_sizes(sample: Dict[str, Any], figsize: tuple = (10, 5)):
    """
    Plot the distribution of nucleus sizes across all tokens in a sample.

    Args:
        sample: Sample dictionary
        figsize: Figure size

    Example:
        plot_nucleus_sizes(sample)
    """
    nucleus_sizes = [len(indices) for indices in sample["top_indices"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(nucleus_sizes, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Nucleus Size")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Nucleus Sizes")
    ax1.axvline(
        np.mean(nucleus_sizes),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(nucleus_sizes):.1f}",
    )
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Over tokens
    ax2.plot(nucleus_sizes, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Nucleus Size")
    ax2.set_title("Nucleus Size Across Tokens")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nNucleus Size Statistics:")
    print(f"  Min: {min(nucleus_sizes)}")
    print(f"  Max: {max(nucleus_sizes)}")
    print(f"  Mean: {np.mean(nucleus_sizes):.2f}")
    print(f"  Median: {np.median(nucleus_sizes):.2f}")
    print(f"  Std: {np.std(nucleus_sizes):.2f}")


# =============================================================================
# Dataset Statistics
# =============================================================================


def compute_dataset_stats(dataset, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute statistics across the entire dataset.

    Args:
        dataset: Dataset object
        max_samples: Maximum number of samples to process (None = all)

    Returns:
        Dictionary with dataset statistics

    Example:
        stats = compute_dataset_stats(ds, max_samples=1000)
        print(f"Avg tokens per sample: {stats['avg_tokens']:.1f}")
    """
    print("📊 Computing dataset statistics...")

    total_samples = 0
    total_tokens = 0
    nucleus_sizes = []

    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        total_samples += 1
        total_tokens += sample["num_tokens"]

        # Check for packed format first
        if "top_indices_low" in sample:
            # For statistics, we only need the length, which is preserved in the low bits structure
            for indices_low in sample["top_indices_low"]:
                nucleus_sizes.append(len(indices_low))
        elif "top_indices" in sample:
            for indices in sample["top_indices"]:
                nucleus_sizes.append(len(indices))

    stats = {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": total_tokens / total_samples
        if total_samples > 0
        else 0,
        "nucleus_size_min": min(nucleus_sizes) if nucleus_sizes else 0,
        "nucleus_size_max": max(nucleus_sizes) if nucleus_sizes else 0,
        "nucleus_size_mean": np.mean(nucleus_sizes) if nucleus_sizes else 0,
        "nucleus_size_median": np.median(nucleus_sizes) if nucleus_sizes else 0,
    }

    print(f"✅ Processed {total_samples} samples with {total_tokens} total tokens")
    return stats


def print_dataset_stats(stats: Dict[str, Any]):
    """
    Pretty-print dataset statistics.

    Args:
        stats: Statistics dictionary from compute_dataset_stats

    Example:
        stats = compute_dataset_stats(ds)
        print_dataset_stats(stats)
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Avg tokens per sample: {stats['avg_tokens_per_sample']:.1f}")
    print(f"\nNucleus sizes:")
    print(f"  Min: {stats['nucleus_size_min']}")
    print(f"  Max: {stats['nucleus_size_max']}")
    print(f"  Mean: {stats['nucleus_size_mean']:.2f}")
    print(f"  Median: {stats['nucleus_size_median']:.2f}")
    print("=" * 60)


# =============================================================================
# Token Decoding (requires tokenizer)
# =============================================================================


def decode_top_tokens(
    sample: Dict[str, Any], token_idx: int, tokenizer, top_k: int = 10
) -> List[tuple]:
    """
    Decode and show the top-k tokens with their probabilities.

    Args:
        sample: Sample dictionary
        token_idx: Token index
        tokenizer: HuggingFace tokenizer
        top_k: Number of top tokens to show

    Returns:
        List of (token_string, probability) tuples

    Example:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

        top_tokens = decode_top_tokens(sample, token_idx=0, tokenizer=tokenizer)
        for token, prob in top_tokens:
            print(f"{token:20s} {prob:.4f}")
    """
    indices, logits = dequantize_top_logits(sample, token_idx)
    lse = sample["logsumexp"][token_idx]
    probs = logits_to_probs(logits, lse)

    # Get top k
    top_k = min(top_k, len(indices))

    results = []
    for i in range(top_k):
        token_str = tokenizer.decode([indices[i]])
        results.append((token_str, probs[i]))

    return results


def print_top_tokens(
    sample: Dict[str, Any], token_idx: int, tokenizer, top_k: int = 10
):
    """
    Print the top-k tokens with their probabilities in a nice format.

    Args:
        sample: Sample dictionary
        token_idx: Token index
        tokenizer: HuggingFace tokenizer
        top_k: Number of top tokens to show

    Example:
        print_top_tokens(sample, token_idx=0, tokenizer=tokenizer, top_k=20)
    """
    tokens = decode_top_tokens(sample, token_idx, tokenizer, top_k)

    print(f"\nTop {len(tokens)} tokens for position {token_idx}:")
    print("-" * 50)
    for i, (token, prob) in enumerate(tokens, 1):
        # Escape special characters for display
        token_display = repr(token)[1:-1]  # Remove outer quotes
        print(f"{i:3d}. {token_display:30s} {prob:8.5f} ({prob * 100:5.2f}%)")
    print("-" * 50)


# =============================================================================
# Quick start helper
# =============================================================================


def quick_explore(
    dataset_id: str,
    sample_idx: int = 0,
    token_idx: int = 0,
    model_id: Optional[str] = None,
):
    """
    Quick exploration of a dataset sample.

    Args:
        dataset_id: HuggingFace dataset ID
        sample_idx: Sample index to explore
        token_idx: Token index to visualize
        model_id: Optional model ID for token decoding

    Example:
        quick_explore("seba/my-logits", sample_idx=0, token_idx=0,
                     model_id="Qwen/Qwen2.5-0.5B-Instruct")
    """
    # Load dataset
    ds = load_logit_dataset(dataset_id)

    # Get sample
    sample = get_sample(ds, sample_idx)
    print_sample_info(sample)

    # Plot visualizations
    plot_nucleus_sizes(sample)
    plot_logit_distribution(sample, token_idx)
    plot_nucleus_vs_sampled(sample, token_idx)

    # Decode tokens if model provided
    if model_id:
        from transformers import AutoTokenizer

        print(f"\n🤖 Loading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print_top_tokens(sample, token_idx, tokenizer, top_k=20)
