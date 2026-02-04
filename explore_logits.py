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
import torch
from datasets import load_dataset, IterableDataset
from typing import Optional, Union, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from huggingface_hub import hf_hub_download
from termcolor import colored
import html
import pyarrow as pa
import pyarrow.dataset as pa_dataset
from dataclasses import dataclass, field
from tqdm.auto import tqdm

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


def load_logit_dataset(dataset_id: str, split: str = "train", streaming: bool = True):
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
    # Check if it's an IterableDataset (streaming)
    # IterableDataset has __getitem__ but it returns a column, not a row by index!
    if isinstance(dataset, IterableDataset):
        item = None
        # Try efficient skip/take if available
        if hasattr(dataset, "skip") and hasattr(dataset, "take"):
            try:
                # take(1) returns an iterable, we need the first item
                item = list(dataset.skip(index).take(1))[0]
            except Exception:
                pass  # Fallback to loop

        if item is None:
            # Fallback to iteration
            for i, curr in enumerate(dataset):
                if i == index:
                    item = curr
                    break
            if item is None:
                raise IndexError(f"Index {index} out of range")
    elif hasattr(dataset, "__getitem__"):
        # Map-style dataset (random access)
        item = dataset[index]
    else:
        # Fallback for generic iterables
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


# =============================================================================
# Advanced Analysis & Alignment
# =============================================================================


def load_extraction_config(repo_id: str) -> Dict[str, Any]:
    """
    Load the extraction configuration from the HuggingFace dataset repository.

    Args:
        repo_id: The ID of the logit dataset (e.g. "user/dataset-logits")

    Returns:
        Configuration dictionary
    """
    try:
        config_path = hf_hub_download(
            repo_id=repo_id, filename="checkpoint.json", repo_type="dataset"
        )
        with open(config_path, "r") as f:
            data = json.load(f)
            return data.get("config", {})
    except Exception as e:
        print(f"⚠️ Could not load config from {repo_id}: {e}")
        return {}


def analyze_token_probabilities(
    logit_sample: Dict[str, Any],
    original_text_sample: Dict[str, Any],
    tokenizer,
    chat_template: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    top_k_predictions: int = 5,
) -> List[Dict[str, Any]]:
    """
    Align extracted logits with the original text to analyze the probability of actual tokens.

    Args:
        logit_sample: Sample from the logit dataset
        original_text_sample: Corresponding sample from the original text dataset
        tokenizer: Tokenizer used for the model
        chat_template: Optional chat template override
        config: Extraction configuration (for exclusion settings)

    Returns:
        List of dicts containing analysis for each token
    """
    config = config or {}

    # 1. Reconstruct inputs to match what the model saw
    # We need to reconstruct the message format expected by the chat template
    # Depending on the dataset, we might need to map columns
    if "messages" in original_text_sample:
        messages = original_text_sample["messages"]
    elif "conversation" in original_text_sample:
        messages = original_text_sample["conversation"]
    elif "conversations" in original_text_sample:
        messages = original_text_sample["conversations"]
    else:
        # Fallback for simple user/assistant columns if known
        # This is a bit heuristical, might need config to know column names
        user_content = original_text_sample.get(
            "question", original_text_sample.get("input", "")
        )
        assistant_content = original_text_sample.get(
            "answer", original_text_sample.get("output", "")
        )
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

    # 2. Tokenize and get assistant mask
    tokenized = tokenizer.apply_chat_template(
        messages,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
        chat_template=chat_template,
    )

    input_ids = tokenized["input_ids"][0]
    assistant_mask = tokenized["assistant_masks"][0].bool()

    # --- Replicate Extraction Filtering Logic ---

    # Define special tokens to exclude
    # This tries to mimic the logic in extract_logits.py

    # Exclude end tokens?
    if config.get(
        "exclude_end_tokens", False
    ):  # Default to True if not in config, safer
        # Try to find end tokens
        exclude_tokens = []
        if tokenizer.pad_token_id is not None:
            exclude_tokens.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            exclude_tokens.append(tokenizer.eos_token_id)

        # Add common special tokens if present in vocab
        for special in ["<|im_end|>", "<|endoftext|>"]:
            if special in tokenizer.get_vocab():
                exclude_tokens.append(tokenizer.get_vocab()[special])

        if exclude_tokens:
            exclude_tensor = torch.tensor(exclude_tokens)
            # Logic: If previous token was special, mask current token?
            # Original script: is_not_after_special = ~torch.isin(batch["input_ids"].roll(1), exclude_tokens)
            # This implies we exclude tokens that FOLLOW a special token (like partial end sequences?)
            # Or maybe it's excluding the special tokens themselves?
            # Wait, `batch["input_ids"].roll(1)` shifts right. So it checks the previous token.
            # If index i has special token at i-1, then roll(1)[i] is special.
            # So it masks i.

            is_not_after_special = ~torch.isin(
                input_ids.roll(1), exclude_tensor.to(input_ids.device)
            )
            # Also mask the special tokens themselves if they are assistant tokens?
            # The original script doesn't explicitly mask the special tokens here,
            # assuming assistant_mask handles it (or they are not part of assistant response)
            # But usually <|im_end|> IS part of assistant response in some templates.

            assistant_mask = assistant_mask & is_not_after_special

    # Exclude think tags?
    if config.get("exclude_think_tags", False):
        # Identify think tokens
        think_token_ids = []
        for tag in [
            "<|thought|>",
            "<|im_start|>thought",
            "<think>",
        ]:  # Common variations
            if tag in tokenizer.get_vocab():
                think_token_ids.append(tokenizer.get_vocab()[tag])

        if think_token_ids:
            think_tensor = torch.tensor(think_token_ids)
            is_not_think = torch.ones_like(input_ids, dtype=torch.bool)

            for think_id in think_token_ids:
                think_id_tensor = torch.tensor([think_id]).to(input_ids.device)

                # Exclude the token itself
                is_not_think &= ~torch.isin(input_ids, think_id_tensor)

                # Exclude a few tokens after (replicating original script loop 1..4)
                for shift in range(1, 4):
                    is_not_think &= ~torch.isin(input_ids.roll(shift), think_id_tensor)

            assistant_mask = assistant_mask & is_not_think

    # 3. Extract assistant tokens (filtered)
    # We want to keep track of ALL assistant tokens, but know which ones were skipped
    # So we get indices of assistant tokens

    # Original assistant mask (raw from template)
    raw_assistant_mask = tokenized["assistant_masks"][0].bool()
    raw_assistant_indices = raw_assistant_mask.nonzero(as_tuple=True)[0]

    # Filtered assistant mask (what was extracted)
    filtered_assistant_indices = assistant_mask.nonzero(as_tuple=True)[0]
    filtered_indices_set = set(filtered_assistant_indices.tolist())

    # Verify alignment
    num_logit_tokens = logit_sample["num_tokens"]
    num_filtered_tokens = len(filtered_assistant_indices)

    if num_logit_tokens != num_filtered_tokens:
        print(
            f"⚠️ Warning: Token count mismatch! Logits: {num_logit_tokens}, Text (Filtered): {num_filtered_tokens}"
        )
        # Try to fallback: if simple alignment works better?
        # Or just allow mismatch and align as much as possible from end?

    # 4. Analyze each token
    analysis = []

    # Iterator for logit index
    logit_idx_counter = 0

    # Iterate over ALL assistant tokens (raw)
    for idx in raw_assistant_indices:
        idx = idx.item()
        token_id = input_ids[idx].item()
        token_str = tokenizer.decode([token_id])

        is_skipped = idx not in filtered_indices_set

        # If we have extracted logits for this token
        if not is_skipped and logit_idx_counter < num_logit_tokens:
            logits_info = dequantize_all_logits(logit_sample, logit_idx_counter)
            logit_idx_counter += 1

            nucleus_indices = logits_info["nucleus_indices"]
            nucleus_logits = logits_info["nucleus_logits"]
            sampled_indices = logits_info["sampled_indices"]
            sampled_logits = logits_info["sampled_logits"]
            lse = logits_info["logsumexp"]

            # Calculate probabilities
            nucleus_probs = logits_to_probs(nucleus_logits, lse)
            sampled_probs = logits_to_probs(sampled_logits, lse)

            # Find where the actual token is
            in_nucleus = token_id in nucleus_indices
            in_sampled = token_id in sampled_indices

            prob = 0.0
            rank = -1

            if in_nucleus:
                loc = np.where(nucleus_indices == token_id)[0][0]
                prob = nucleus_probs[loc]
                rank = loc
            elif in_sampled:
                loc = np.where(sampled_indices == token_id)[0][0]
                prob = sampled_probs[loc]
                rank = len(nucleus_indices) + loc
            else:
                prob = 0.0
                rank = ">" + str(len(nucleus_indices) + len(sampled_indices))

            # --- Extract Top K Predictions ---
            # Combine nucleus and sampled to find top tokens
            # Nucleus is sorted by prob, sampled might not be strictly sorted globally but usually is

            # Create a unified list of (token_id, prob)
            all_candidates = []
            for tid, p in zip(nucleus_indices, nucleus_probs):
                all_candidates.append((tid, p))
            for tid, p in zip(sampled_indices, sampled_probs):
                all_candidates.append((tid, p))

            # Sort by probability descending
            all_candidates.sort(key=lambda x: x[1], reverse=True)

            # Take top K
            top_predictions = []
            for tid, p in all_candidates[:top_k_predictions]:
                t_str = tokenizer.decode([tid])
                top_predictions.append(
                    {"token_id": int(tid), "token": t_str, "prob": float(p)}
                )

            analysis.append(
                {
                    "token_id": token_id,
                    "token": token_str,
                    "prob": float(prob),
                    "rank": rank,
                    "in_nucleus": bool(in_nucleus),
                    "in_sampled": bool(in_sampled),
                    "skipped": False,
                    "top_predictions": top_predictions,
                }
            )
        else:
            # Skipped or ran out of logits
            analysis.append(
                {
                    "token_id": token_id,
                    "token": token_str,
                    "prob": 0.0,
                    "rank": -1,
                    "in_nucleus": False,
                    "in_sampled": False,
                    "skipped": True,
                }
            )

    return analysis


def visualize_conversation_probabilities(
    analysis: List[Dict[str, Any]], return_html: bool = False, detailed: bool = True
):
    """
    Visualize the conversation with tokens colored by probability.

    Args:
        analysis: Result from analyze_token_probabilities
        return_html: If True, return HTML string instead of printing
        detailed: If True, print detailed line-by-line info (Console only)

    Returns:
        None or HTML string
    """

    def get_color(prob, skipped=False):
        if skipped:
            return "white"
        # Green (high) -> Yellow (med) -> Red (low)
        if prob > 0.9:
            return "green"
        if prob > 0.5:
            return "cyan"
        if prob > 0.1:
            return "yellow"
        if prob > 0.01:
            return "red"
        return "magenta"  # Very low

    def get_html_color(prob, skipped=False):
        if skipped:
            return "#888888"
        if prob > 0.9:
            return "#4CAF50"  # Green
        if prob > 0.5:
            return "#CDDC39"  # Lime
        if prob > 0.1:
            return "#FFC107"  # Amber
        if prob > 0.01:
            return "#FF5722"  # Deep Orange
        return "#795548"  # Brown (very low)

    # HTML output (for Jupyter) - primarily for quick visual
    if return_html:
        html_parts = [
            '<div style="font-family: monospace; line-height: 1.5; background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px;">'
        ]

        for item in analysis:
            token = item["token"]
            # Escape HTML special chars
            token = html.escape(token)
            # Visualize newlines
            token = token.replace("\n", "<br/>")

            prob = item["prob"]
            rank = item["rank"]
            skipped = item.get("skipped", False)
            color = get_html_color(prob, skipped)

            tooltip = f"Token: {html.escape(repr(item['token']))} | Prob: {prob:.4f} | Rank: {rank}"
            if skipped:
                tooltip = "Skipped (Think tag or End token)"
                style = f"color: {color}; text-decoration: line-through; border-bottom: 1px dotted #555;"
            else:
                style = f"color: {color}; border-bottom: 1px dotted #555;"

            span = f'<span style="{style}" title="{tooltip}">{token}</span>'
            html_parts.append(span)

        html_parts.append("</div>")

        # Add legend
        html_parts.append("""
        <div style="margin-top: 10px; font-size: 0.8em; color: #888;">
            Legend: 
            <span style="color: #4CAF50"> >90% </span> |
            <span style="color: #CDDC39"> >50% </span> |
            <span style="color: #FFC107"> >10% </span> |
            <span style="color: #FF5722"> >1% </span> |
            <span style="color: #795548"> <1% </span> |
            <span style="color: #888888; text-decoration: line-through;"> Skipped </span>
        </div>
        """)

        return "".join(html_parts)

    # Console output
    if not detailed:
        print("\nToken Probabilities:")
        print("-" * 60)
        line_buffer = ""
        for item in analysis:
            token = item["token"]
            prob = item["prob"]
            skipped = item.get("skipped", False)
            color = get_color(prob, skipped)

            if "\n" in token:
                print(line_buffer)
                print(colored("↵", "grey"))  # Visual newline
                line_buffer = ""
                token = token.replace("\n", "")

            token_display = repr(token)[1:-1]
            if not token_display:
                continue

            try:
                if skipped:
                    print(colored(token_display, color, attrs=["strike"]), end=" ")
                else:
                    print(colored(token_display, color), end=" ")
            except:
                print(token_display, end=" ")
        print("\n" + "-" * 60)
    else:
        # DETAILED LINE-BY-LINE OUTPUT
        print("\nDetailed Token Analysis:")
        print("=" * 120)
        print(f"{'Real Token':<30} | {'Extracted Predictions (Top 5)':<80}")
        print("-" * 120)

        for item in analysis:
            token = item["token"]
            skipped = item.get("skipped", False)
            top_predictions = item.get("top_predictions", [])

            # Format real token
            token_display = repr(token)
            if len(token_display) > 28:
                token_display = token_display[:25] + "..."

            if skipped:
                token_colored = colored(token_display, "white", attrs=["dark"])
                predictions_str = colored("(Skipped - Filtered)", "grey")
            else:
                prob = item["prob"]
                rank = item["rank"]
                color = get_color(prob)
                token_colored = colored(token_display, color)

                # Format predictions
                preds_parts = []
                for pred in top_predictions:
                    p_token = pred["token"]
                    p_prob = pred["prob"]
                    p_color = get_color(p_prob)

                    p_display = repr(p_token)
                    if len(p_display) > 15:
                        p_display = p_display[:12] + "..."

                    # Highlight if it matches real token
                    # Note: We compare token IDs ideally, but here we just have strings and IDs in dict
                    # The real token ID is item["token_id"]
                    is_match = pred["token_id"] == item["token_id"]

                    pred_str = f"{p_display}({p_prob:.2f})"
                    if is_match:
                        pred_str = colored(
                            pred_str, p_color, attrs=["bold", "underline"]
                        )
                    else:
                        pred_str = colored(pred_str, p_color)

                    preds_parts.append(pred_str)

                predictions_str = ", ".join(preds_parts)

            print(f"{token_colored:<30} | {predictions_str}")

        print("=" * 120)


# =============================================================================
# Nucleus Size Analysis (Streaming)
# =============================================================================


@dataclass
class NucleusSizeStats:
    """Statistics about nucleus sizes across a dataset."""

    # Per-token statistics
    all_sizes: np.ndarray  # All nucleus sizes (can be large!)
    total_tokens: int
    total_samples: int

    # Aggregated statistics
    min_size: int
    max_size: int
    mean_size: float
    median_size: float
    std_size: float
    percentiles: Dict[
        int, float
    ]  # {5: val, 25: val, 50: val, 75: val, 95: val, 99: val}

    # Size distribution (histogram-ready)
    size_counts: Dict[int, int]  # {size: count}

    # Per-sample statistics
    samples_mean_sizes: np.ndarray  # Mean nucleus size per sample
    samples_token_counts: np.ndarray  # Token count per sample

    def __repr__(self):
        return (
            f"NucleusSizeStats(\n"
            f"  total_samples={self.total_samples:,},\n"
            f"  total_tokens={self.total_tokens:,},\n"
            f"  min={self.min_size}, max={self.max_size},\n"
            f"  mean={self.mean_size:.2f}, median={self.median_size:.1f}, std={self.std_size:.2f},\n"
            f"  p5={self.percentiles[5]:.1f}, p95={self.percentiles[95]:.1f}, p99={self.percentiles[99]:.1f}\n"
            f")"
        )

    def _repr_html_(self):
        """Rich HTML representation for Jupyter notebooks."""
        return f"""
        <div style="font-family: monospace; background: #f5f5f5; padding: 15px; border-radius: 8px;">
            <h3 style="margin-top: 0;">Nucleus Size Statistics</h3>
            <table style="border-collapse: collapse;">
                <tr><td style="padding: 4px 12px;"><b>Total Samples</b></td><td>{self.total_samples:,}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>Total Tokens</b></td><td>{self.total_tokens:,}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>Min Size</b></td><td>{self.min_size}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>Max Size</b></td><td>{self.max_size}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>Mean</b></td><td>{self.mean_size:.2f}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>Median</b></td><td>{self.median_size:.1f}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>Std Dev</b></td><td>{self.std_size:.2f}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>P5</b></td><td>{self.percentiles[5]:.1f}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>P25</b></td><td>{self.percentiles[25]:.1f}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>P75</b></td><td>{self.percentiles[75]:.1f}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>P95</b></td><td>{self.percentiles[95]:.1f}</td></tr>
                <tr><td style="padding: 4px 12px;"><b>P99</b></td><td>{self.percentiles[99]:.1f}</td></tr>
            </table>
        </div>
        """


def analyze_nucleus_sizes(
    dataset_id: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    batch_size: int = 1000,
    collect_all_sizes: bool = True,
    show_progress: bool = True,
    device: Optional[str] = None,
) -> NucleusSizeStats:
    """
    Analyze the distribution of nucleus sizes (extracted logits per token) across a dataset.

    Uses streaming with optimizations from extract_logits.py for efficient processing
    of large datasets without loading everything into memory.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "user/logits-dataset")
        split: Dataset split to analyze (default: "train")
        max_samples: Maximum number of samples to process (None = all)
        batch_size: Batch size for streaming iteration (higher = faster but more memory)
        collect_all_sizes: If True, collect all individual sizes for detailed analysis.
                          Set to False for very large datasets to save memory.
        show_progress: Whether to show a progress bar
        device: PyTorch device for GPU acceleration (e.g., "cuda", "mps", or None for CPU/numpy).
                When set, uses PyTorch tensors on the specified device for faster computation.

    Returns:
        NucleusSizeStats object with comprehensive statistics

    Example (Jupyter notebook):
        ```python
        from explore_logits import analyze_nucleus_sizes, plot_nucleus_size_analysis

        # Analyze the dataset (CPU)
        stats = analyze_nucleus_sizes("user/my-logits", max_samples=10000)

        # Analyze with GPU acceleration
        stats = analyze_nucleus_sizes("user/my-logits", max_samples=10000, device="cuda")

        # Display stats (rich HTML in Jupyter)
        display(stats)

        # Visualize
        plot_nucleus_size_analysis(stats)
        ```
    """
    import time

    print(f"📊 Analyzing nucleus sizes in: {dataset_id}")

    # Timing accumulators
    timings = {
        "batch_iteration": 0.0,
        "data_conversion": 0.0,
        "gpu_transfer": 0.0,
        "gpu_compute": 0.0,
        "per_sample_stats": 0.0,
        "histogram": 0.0,
        "collect_sizes": 0.0,
    }

    # Set up device for GPU acceleration
    use_gpu = device is not None
    if use_gpu:
        torch_device = torch.device(device)
        print(f"🚀 Using PyTorch acceleration on device: {torch_device}")

    t_setup_start = time.perf_counter()

    # OPTIMIZATION: Enable prefetching and larger buffer for faster streaming
    # (Same optimization used in extract_logits.py get_resume_info)
    fragment_scan_options = pa_dataset.ParquetFragmentScanOptions(
        cache_options=pa.CacheOptions(
            prefetch_limit=10_000,  # Prefetch next chunk while processing current
            range_size_limit=128 << 20,  # 128 MiB block size
        ),
    )

    # OPTIMIZATION: Only load the columns we need
    ds = load_dataset(
        dataset_id,
        split=split,
        streaming=True,
        columns=["top_counts", "num_tokens"],
        fragment_scan_options=fragment_scan_options,
    )

    print(f"⏱️  Dataset setup: {time.perf_counter() - t_setup_start:.3f}s")

    # Collectors
    all_sizes_list = [] if collect_all_sizes else None
    size_counts: Dict[int, int] = defaultdict(int)
    samples_mean_sizes = []
    samples_token_counts = []

    total_tokens = 0
    total_samples = 0

    # Running stats for when we don't collect all sizes
    running_sum = 0
    running_sum_sq = 0
    global_min = float("inf")
    global_max = float("-inf")

    # OPTIMIZATION: Batch iteration instead of row-by-row (~10x faster)
    iterator = ds.batch(batch_size=batch_size)

    # Progress bar: show sample count with total if max_samples is set
    pbar = None
    if show_progress:
        pbar = tqdm(
            total=max_samples,
            desc="Analyzing nucleus sizes",
            unit="samples",
            dynamic_ncols=True,
        )

    t_loop_start = time.perf_counter()
    t_batch_start = time.perf_counter()

    for batch in iterator:
        timings["batch_iteration"] += time.perf_counter() - t_batch_start

        batch_top_counts = batch["top_counts"]
        batch_num_tokens = batch["num_tokens"]
        batch_samples_processed = 0

        if use_gpu:
            # GPU-accelerated batch processing
            # Collect all counts from this batch and process together
            batch_counts_list = []
            batch_lengths = []
            batch_valid_indices = []

            for i, (sample_counts, num_tokens) in enumerate(
                zip(batch_top_counts, batch_num_tokens)
            ):
                if (
                    max_samples is not None
                    and total_samples + len(batch_valid_indices) >= max_samples
                ):
                    break
                if len(sample_counts) == 0:
                    continue
                batch_counts_list.append(sample_counts)
                batch_lengths.append(len(sample_counts))
                batch_valid_indices.append(i)

            if batch_counts_list:
                # Concatenate all counts and move to GPU
                t0 = time.perf_counter()
                all_batch_counts = np.concatenate(
                    [np.array(c, dtype=np.int32) for c in batch_counts_list]
                )
                timings["data_conversion"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                counts_tensor = torch.from_numpy(all_batch_counts).to(torch_device)
                if torch_device.type == "cuda":
                    torch.cuda.synchronize()
                elif torch_device.type == "mps":
                    torch.mps.synchronize()
                timings["gpu_transfer"] += time.perf_counter() - t0

                # Compute global stats on GPU
                t0 = time.perf_counter()
                batch_min = counts_tensor.min().item()
                batch_max = counts_tensor.max().item()
                batch_sum = counts_tensor.sum().item()
                batch_sum_sq = (counts_tensor.float() ** 2).sum().item()
                timings["gpu_compute"] += time.perf_counter() - t0

                global_min = min(global_min, batch_min)
                global_max = max(global_max, batch_max)
                running_sum += batch_sum
                running_sum_sq += batch_sum_sq

                # Compute per-sample means on GPU using segment reduction
                t0 = time.perf_counter()
                lengths_tensor = torch.tensor(batch_lengths, device=torch_device)
                offsets = torch.cat(
                    [
                        torch.tensor([0], device=torch_device),
                        lengths_tensor.cumsum(0)[:-1],
                    ]
                )

                for idx, length in enumerate(batch_lengths):
                    start = offsets[idx].item()
                    end = start + length
                    sample_mean = counts_tensor[start:end].float().mean().item()
                    samples_mean_sizes.append(sample_mean)
                    samples_token_counts.append(length)
                timings["per_sample_stats"] += time.perf_counter() - t0

                # Histogram: use torch.unique on GPU, then update dict on CPU
                t0 = time.perf_counter()
                unique_vals, unique_cnts = torch.unique(
                    counts_tensor, return_counts=True
                )
                unique_vals = unique_vals.cpu().numpy()
                unique_cnts = unique_cnts.cpu().numpy()
                for size, cnt in zip(unique_vals, unique_cnts):
                    size_counts[int(size)] += int(cnt)
                timings["histogram"] += time.perf_counter() - t0

                # Collect sizes if needed (keep on CPU as numpy)
                t0 = time.perf_counter()
                if collect_all_sizes:
                    offset = 0
                    counts_np = all_batch_counts
                    for length in batch_lengths:
                        all_sizes_list.append(
                            counts_np[offset : offset + length].copy()
                        )
                        offset += length
                timings["collect_sizes"] += time.perf_counter() - t0

                total_tokens += len(all_batch_counts)
                total_samples += len(batch_lengths)
                batch_samples_processed = len(batch_lengths)
        else:
            # CPU/numpy processing (original code path)
            for sample_counts, num_tokens in zip(batch_top_counts, batch_num_tokens):
                if max_samples is not None and total_samples >= max_samples:
                    break

                # Convert to numpy for efficient processing
                t0 = time.perf_counter()
                counts = np.array(sample_counts, dtype=np.int32)
                timings["data_conversion"] += time.perf_counter() - t0

                if len(counts) == 0:
                    continue

                # Per-sample stats
                t0 = time.perf_counter()
                sample_mean = counts.mean()
                samples_mean_sizes.append(sample_mean)
                samples_token_counts.append(len(counts))

                sample_min = counts.min()
                sample_max = counts.max()
                global_min = min(global_min, sample_min)
                global_max = max(global_max, sample_max)

                running_sum += counts.sum()
                running_sum_sq += (counts.astype(np.float64) ** 2).sum()
                timings["per_sample_stats"] += time.perf_counter() - t0

                # Update global stats
                total_tokens += len(counts)
                total_samples += 1
                batch_samples_processed += 1

                # Collect sizes
                t0 = time.perf_counter()
                if collect_all_sizes:
                    all_sizes_list.append(counts)
                timings["collect_sizes"] += time.perf_counter() - t0

                # Update histogram
                t0 = time.perf_counter()
                unique, unique_counts = np.unique(counts, return_counts=True)
                for size, cnt in zip(unique, unique_counts):
                    size_counts[size] += cnt
                timings["histogram"] += time.perf_counter() - t0

        # Update progress bar after each batch
        if pbar is not None:
            pbar.update(batch_samples_processed)
            pbar.set_postfix(
                tokens=f"{total_tokens:,}",
                avg=f"{running_sum / total_tokens:.1f}" if total_tokens > 0 else "N/A",
            )

        if max_samples is not None and total_samples >= max_samples:
            break

        t_batch_start = time.perf_counter()

    if pbar is not None:
        pbar.close()

    # Print timing breakdown
    total_time = time.perf_counter() - t_loop_start
    print(f"\n⏱️  Timing breakdown (total loop: {total_time:.2f}s):")
    for name, t in sorted(timings.items(), key=lambda x: -x[1]):
        pct = (t / total_time * 100) if total_time > 0 else 0
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {name:20s}: {t:7.2f}s ({pct:5.1f}%) {bar}")

    if total_samples == 0:
        raise ValueError("No samples found in dataset")

    # Compute final statistics
    if collect_all_sizes and all_sizes_list:
        all_sizes = np.concatenate(all_sizes_list)
        mean_size = all_sizes.mean()
        std_size = all_sizes.std()
        median_size = np.median(all_sizes)
        percentiles = {p: np.percentile(all_sizes, p) for p in [5, 25, 50, 75, 95, 99]}
    else:
        # Compute from running stats
        all_sizes = np.array([], dtype=np.int32)
        mean_size = running_sum / total_tokens if total_tokens > 0 else 0
        variance = (
            (running_sum_sq / total_tokens) - (mean_size**2) if total_tokens > 0 else 0
        )
        std_size = np.sqrt(max(0, variance))

        # Estimate percentiles from histogram
        sorted_sizes = sorted(size_counts.keys())
        cumsum = 0
        percentiles = {}
        for p in [5, 25, 50, 75, 95, 99]:
            target = total_tokens * p / 100
            for size in sorted_sizes:
                cumsum += size_counts[size]
                if cumsum >= target and p not in percentiles:
                    percentiles[p] = float(size)
                    break
            if p not in percentiles:
                percentiles[p] = float(sorted_sizes[-1]) if sorted_sizes else 0

        median_size = percentiles[50]

    print(f"✅ Analyzed {total_samples:,} samples with {total_tokens:,} tokens")

    return NucleusSizeStats(
        all_sizes=all_sizes,
        total_tokens=total_tokens,
        total_samples=total_samples,
        min_size=int(global_min) if global_min != float("inf") else 0,
        max_size=int(global_max) if global_max != float("-inf") else 0,
        mean_size=float(mean_size),
        median_size=float(median_size),
        std_size=float(std_size),
        percentiles=percentiles,
        size_counts=dict(size_counts),
        samples_mean_sizes=np.array(samples_mean_sizes),
        samples_token_counts=np.array(samples_token_counts),
    )


def plot_nucleus_size_analysis(
    stats: NucleusSizeStats,
    figsize: tuple = (14, 10),
    max_size_display: Optional[int] = None,
):
    """
    Create comprehensive visualizations of nucleus size statistics.

    Best viewed in a Jupyter notebook for interactive exploration.

    Args:
        stats: NucleusSizeStats object from analyze_nucleus_sizes()
        figsize: Figure size (width, height)
        max_size_display: Maximum nucleus size to display in histograms (None = auto)

    Example:
        ```python
        stats = analyze_nucleus_sizes("user/my-logits")
        plot_nucleus_size_analysis(stats)
        ```
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Determine max size for display
    if max_size_display is None:
        max_size_display = min(stats.max_size, int(stats.percentiles[99] * 1.5))

    # 1. Histogram of nucleus sizes (from size_counts)
    ax1 = axes[0, 0]
    sizes = sorted([s for s in stats.size_counts.keys() if s <= max_size_display])
    counts = [stats.size_counts[s] for s in sizes]

    ax1.bar(
        sizes, counts, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5
    )
    ax1.axvline(
        stats.mean_size,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {stats.mean_size:.1f}",
    )
    ax1.axvline(
        stats.median_size,
        color="orange",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {stats.median_size:.1f}",
    )
    ax1.set_xlabel("Nucleus Size (logits per token)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Nucleus Sizes")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xlim(0, max_size_display + 1)

    # 2. Log-scale histogram for better tail visibility
    ax2 = axes[0, 1]
    ax2.bar(sizes, counts, color="coral", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.axvline(
        stats.percentiles[95],
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"P95: {stats.percentiles[95]:.0f}",
    )
    ax2.axvline(
        stats.percentiles[99],
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"P99: {stats.percentiles[99]:.0f}",
    )
    ax2.set_xlabel("Nucleus Size (logits per token)")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Nucleus Size Distribution (Log Scale)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xlim(0, max_size_display + 1)

    # 3. Per-sample mean nucleus size distribution
    ax3 = axes[1, 0]
    ax3.hist(
        stats.samples_mean_sizes,
        bins=50,
        color="mediumseagreen",
        alpha=0.7,
        edgecolor="black",
    )
    ax3.axvline(
        stats.samples_mean_sizes.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean of means: {stats.samples_mean_sizes.mean():.1f}",
    )
    ax3.set_xlabel("Mean Nucleus Size per Sample")
    ax3.set_ylabel("Number of Samples")
    ax3.set_title("Distribution of Per-Sample Mean Nucleus Sizes")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # 4. Tokens per sample vs mean nucleus size scatter
    ax4 = axes[1, 1]
    # Subsample if too many points
    max_points = 5000
    if len(stats.samples_mean_sizes) > max_points:
        idx = np.random.choice(len(stats.samples_mean_sizes), max_points, replace=False)
        x = stats.samples_token_counts[idx]
        y = stats.samples_mean_sizes[idx]
        alpha = 0.3
    else:
        x = stats.samples_token_counts
        y = stats.samples_mean_sizes
        alpha = 0.5

    ax4.scatter(x, y, alpha=alpha, s=10, c="steelblue")
    ax4.set_xlabel("Tokens per Sample")
    ax4.set_ylabel("Mean Nucleus Size")
    ax4.set_title("Tokens vs Mean Nucleus Size per Sample")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("NUCLEUS SIZE SUMMARY")
    print("=" * 60)
    print(f"Total samples: {stats.total_samples:,}")
    print(f"Total tokens:  {stats.total_tokens:,}")
    print(f"Avg tokens/sample: {stats.total_tokens / stats.total_samples:.1f}")
    print(f"\nNucleus size statistics:")
    print(f"  Min:    {stats.min_size}")
    print(f"  Max:    {stats.max_size}")
    print(f"  Mean:   {stats.mean_size:.2f}")
    print(f"  Median: {stats.median_size:.1f}")
    print(f"  Std:    {stats.std_size:.2f}")
    print(f"\nPercentiles:")
    print(f"  P5:  {stats.percentiles[5]:.1f}")
    print(f"  P25: {stats.percentiles[25]:.1f}")
    print(f"  P50: {stats.percentiles[50]:.1f}")
    print(f"  P75: {stats.percentiles[75]:.1f}")
    print(f"  P95: {stats.percentiles[95]:.1f}")
    print(f"  P99: {stats.percentiles[99]:.1f}")

    # Estimate storage savings
    avg_nucleus = stats.mean_size
    max_nucleus = 100  # Default max from extract_logits.py
    savings_pct = (1 - avg_nucleus / max_nucleus) * 100
    print(f"\nStorage efficiency:")
    print(f"  Avg nucleus size: {avg_nucleus:.1f} (max: {max_nucleus})")
    print(f"  Storage saved: ~{savings_pct:.1f}% vs fixed-size extraction")
    print("=" * 60)


def compute_bytes_per_token_stats(
    stats: NucleusSizeStats, sampled_n: int = 24
) -> Dict[str, float]:
    """
    Compute storage bytes per token based on nucleus size statistics.

    Args:
        stats: NucleusSizeStats from analyze_nucleus_sizes()
        sampled_n: Number of sampled logits (default: 24)

    Returns:
        Dictionary with bytes per token statistics

    Example:
        ```python
        stats = analyze_nucleus_sizes("user/my-logits")
        bytes_stats = compute_bytes_per_token_stats(stats)
        print(f"Average bytes/token: {bytes_stats['mean']:.1f}")
        ```
    """

    def bytes_for_nucleus_size(n: int) -> int:
        """Calculate bytes for a given nucleus size."""
        # Nucleus data
        nucleus_bytes = (
            n * 2  # indices_low (uint16)
            + (n + 3) // 4  # indices_high (packed 2-bit, ceil division)
            + n * 2  # logits_quantized (uint16)
            + 1  # count (uint8)
            + 4  # min (float32)
            + 4  # max (float32)
        )

        # Sampled data (fixed size)
        sampled_bytes = (
            sampled_n * 2  # indices_low
            + (sampled_n + 3) // 4  # indices_high
            + sampled_n * 2  # logits_quantized
            + 4  # min
            + 4  # max
        )

        # Other
        other_bytes = 8  # logsumexp (float64)

        return nucleus_bytes + sampled_bytes + other_bytes

    # Compute for percentiles and mean
    mean_bytes = bytes_for_nucleus_size(int(round(stats.mean_size)))
    median_bytes = bytes_for_nucleus_size(int(round(stats.median_size)))
    min_bytes = bytes_for_nucleus_size(stats.min_size)
    max_bytes = bytes_for_nucleus_size(stats.max_size)
    p95_bytes = bytes_for_nucleus_size(int(round(stats.percentiles[95])))
    p99_bytes = bytes_for_nucleus_size(int(round(stats.percentiles[99])))

    # Weighted average from histogram
    if stats.size_counts:
        total_bytes = sum(
            bytes_for_nucleus_size(size) * count
            for size, count in stats.size_counts.items()
        )
        weighted_mean_bytes = total_bytes / stats.total_tokens
    else:
        weighted_mean_bytes = mean_bytes

    return {
        "mean": weighted_mean_bytes,
        "median": median_bytes,
        "min": min_bytes,
        "max": max_bytes,
        "p95": p95_bytes,
        "p99": p99_bytes,
        "fixed_100": bytes_for_nucleus_size(100),  # For comparison
    }


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

        print(f"\\n🤖 Loading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print_top_tokens(sample, token_idx, tokenizer, top_k=20)


def execute_analysis(
    logit_dataset_id: str,
    sample_index: int = 0,
    tokenizer_id: Optional[str] = None,
    original_dataset_id: Optional[str] = None,
    chat_template: Optional[str] = None,
):
    """
    Execute full analysis for a sample, aligning logits with original text.

    Args:
        logit_dataset_id: ID of the logit dataset
        sample_index: Index of the sample to analyze
        tokenizer_id: ID of the tokenizer (optional, will try to load from config)
        original_dataset_id: ID of original dataset (optional, will try to load from config)
    """
    # 1. Load config to find original dataset and model
    print(f"🔧 Loading config from {logit_dataset_id}...")
    config = load_extraction_config(logit_dataset_id)

    # 2. Determine Original Dataset and Model
    orig_ds_id = original_dataset_id or config.get("dataset_id")
    if not orig_ds_id:
        raise ValueError(
            "Could not determine original dataset ID from config. Please provide `original_dataset_id`."
        )

    model_id = tokenizer_id or config.get("model_id")
    if not model_id:
        raise ValueError(
            "Could not determine model ID from config. Please provide `tokenizer_id`."
        )

    print(f"📊 Original Dataset: {orig_ds_id}")
    print(f"🤖 Model/Tokenizer: {model_id}")

    # 3. Load Resources
    # Load logit dataset (streaming)
    ds_logits = load_logit_dataset(logit_dataset_id, streaming=True)
    logit_sample = get_sample(ds_logits, sample_index)

    # Load original dataset (streaming)
    print(f"📦 Loading original dataset: {orig_ds_id}")
    ds_orig = load_dataset(orig_ds_id, split="train", streaming=True)

    # Get the original sample (need to match index)
    # Note: If logit dataset has 'index' column matching original dataset, use that.
    # Otherwise assume 1:1 mapping if not shuffled.
    original_idx = logit_sample.get("index", sample_index)
    print(
        f"📍 Fetching sample index: {original_idx} (Logit Sample Index: {sample_index})"
    )

    # In streaming mode, we iterate to find the index
    # Optimization: if original_idx is large, this might be slow, but safe for streaming
    original_sample = None
    # We might need to iterate quite a bit if indices are sparse or shuffled
    # Ideally logic_sample['index'] should correspond to the i-th element of original dataset if not skipped
    # But often it refers to the index column in original dataset.
    # Let's assume for now we can select by index or iterate.
    # Since streaming datasets don't support .iloc, we proceed by iteration

    # Iterate and find matching index
    # Only iterate up to a reasonable limit or if indices match
    # For now, simplistic approach: iterate until we hit the count
    current_idx = 0
    for item in ds_orig:
        if current_idx == original_idx:  # This assumes dataset order is preserved 1:1
            original_sample = item
            break
        current_idx += 1
        # Safety break if we go way past
        if current_idx > original_idx + 1000:
            break

    if original_sample is None:
        print(
            "⚠️ Could not find corresponding sample in original dataset (by iteration). Trying to just take the nth element."
        )
        # Reset and take nth
        ds_orig = load_dataset(orig_ds_id, split="train", streaming=True)
        for i, item in enumerate(ds_orig):
            if i == original_idx:
                original_sample = item
                break

    if original_sample is None:
        raise ValueError(f"Could not find sample {original_idx} in original dataset")

    # Load tokenizer
    print("Pre-loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 4. Run Analysis
    print("✨ Analyzing...")
    analysis = analyze_token_probabilities(
        logit_sample,
        original_sample,
        tokenizer,
        chat_template=chat_template or config.get("chat_template_file"),
        config=config,
    )

    # 5. Visualize
    visualize_conversation_probabilities(analysis)

    return analysis
