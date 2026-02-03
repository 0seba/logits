"""
Compute Dataset Lengths

This script calculates the tokenized length of conversations in a dataset which is critical
for efficient batching in the logit extraction process. It generates a lightweight dataset
containing just the lengths (and optionally indices) that can be used to sort the main
dataset before processing.

Features:
- Optimized parallel processing using HuggingFace fast tokenizers
- Support for custom chat templates
- Streaming mode for massive datasets
- efficient batch processing
- Direct upload to HuggingFace Hub

Usage:
    python compute_lengths.py --model Qwen/Qwen3-4B-Instruct \
                              --dataset MegaScience/MegaScience \
                              --output-repo seba/MegaScience-Lengths \
                              --num-proc 16
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset, Features, Value
from huggingface_hub import HfApi, create_repo
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(description="Compute token lengths for a dataset")

    # Model and Datasest
    parser.add_argument(
        "--model", type=str, required=True, help="Model ID for tokenizer"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Input dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--subset", type=str, default=None, help="Dataset configuration/subset name"
    )

    # Columns
    parser.add_argument(
        "--user-col", type=str, default="question", help="User column name"
    )
    parser.add_argument(
        "--assistant-col", type=str, default="answer", help="Assistant column name"
    )

    # Output
    parser.add_argument(
        "--output-repo", type=str, default=None, help="HuggingFace repo to upload to"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lengths",
        help="Local directory to save to",
    )
    parser.add_argument(
        "--filename", type=str, default="lengths.parquet", help="Output filename"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Upload result to HF Hub"
    )

    # Processing
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for mapping"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=os.cpu_count(),
        help="Number of processes for parallelism",
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Stream dataset instead of downloading"
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Path to custom chat template file",
    )

    return parser.parse_args()


def get_transform_fn(tokenizer, user_col, assistant_col, chat_template=None):
    """Returns the transformation function for batch processing."""

    def transform(batch):
        # Construct conversations
        # batch is a dict of lists: {col: [val1, val2, ...]}
        batch_size = len(batch[user_col])
        conversations = []

        for i in range(batch_size):
            msgs = [
                {"role": "user", "content": batch[user_col][i]},
                {"role": "assistant", "content": batch[assistant_col][i]},
            ]
            conversations.append(msgs)

        # Apply template and tokenize
        # Using return_length=True is supported in some versions, but we'll do it explicitly
        # to ensure compatibility and correctness with apply_chat_template

        # Note: We don't need to generate, just tokenize full sequence
        try:
            # Optimize: apply_chat_template can handle lists of conversations in recent versions
            # But to be safe and support progress bars well in map, we utilize the tokenizer's batch encoding

            # We first format with template (string level) to avoid some tokenizer overhead if possible,
            # but apply_chat_template handles both formatting and tokenization.
            # Ideally we pass 'tokenize=True' (default)

            encodings = tokenizer.apply_chat_template(
                conversations,
                chat_template=chat_template,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,  # Returns list of lists
            )

            lengths = [len(ids) for ids in encodings]

            return {"length": lengths}

        except Exception as e:
            # Fallback for older transformers without batch support in apply_chat_template
            # or other errors
            # print(f"Warning: Batch processing error, falling back to sequential: {e}")
            lengths = []
            for conv in conversations:
                ids = tokenizer.apply_chat_template(
                    conv,
                    chat_template=chat_template,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                lengths.append(len(ids))
            return {"length": lengths}

    return transform


def main():
    args = parse_args()

    print(f"🚀 Starting Length Computation")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    custom_template = None
    if args.chat_template:
        with open(args.chat_template, "r") as f:
            custom_template = f.read()
        print(f"Loaded custom chat template from {args.chat_template}")

    # 2. Load Dataset
    print(f"Loading dataset (streaming={args.streaming})...")
    ds = load_dataset(
        args.dataset, split=args.split, name=args.subset, streaming=args.streaming
    )

    transform_fn = get_transform_fn(
        tokenizer, args.user_col, args.assistant_col, chat_template=custom_template
    )

    # 3. Process
    start_time = time.time()

    if args.streaming:
        print("Processing in streaming mode...")
        # For streaming, we iterate and process in batches manually or via map
        # map() on iterable dataset applies function lazily
        mapped_ds = ds.map(transform_fn, batched=True, batch_size=args.batch_size)

        # We need to realize the results to save them
        # We'll stream to a parquet file directly to avoid memory issues
        output_path = os.path.join(args.output_dir, args.filename)
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"Streaming results to {output_path}...")

        # We define a generator that yields the computed lengths
        def result_gen():
            for row in tqdm(mapped_ds, desc="Computing lengths"):
                yield {"length": row["length"]}

        # Create a new dataset from generator (caches to arrow/parquet)
        # Note: Dataset.from_generator might try to fit in memory, so we better write chunks
        # But simplest way compatible with 'push_to_hub' is generating a Dataset object.
        # If it's too big, we should use lower level arrow streaming.

        # Let's collect lists of lengths. Assuming 10M rows * 4 bytes = 40MB. It fits in RAM.
        # Even 1B rows is 4GB. Most datasets are smaller.

        lengths = []
        for row in tqdm(mapped_ds, desc="Streaming & Computing"):
            lengths.append(row["length"])

        result_ds = Dataset.from_dict({"length": lengths})

    else:
        print(f"Processing in parallel mode (num_proc={args.num_proc})...")
        # For standard datasets, map works great parallelized

        # Only keep necessary columns to reduce memory overhead during map if possible,
        # but map(..., remove_columns=...) is easier
        cols_to_remove = ds.column_names

        result_ds = ds.map(
            transform_fn,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=cols_to_remove,
            desc="Computing lengths",
        )

    end_time = time.time()
    print(f"✅ Computation complete in {end_time - start_time:.2f}s")
    print(f"Total samples: {len(result_ds)}")

    # 4. Save Locally
    os.makedirs(args.output_dir, exist_ok=True)
    local_path = os.path.join(args.output_dir, args.filename)

    # Save as parquet using pyarrow for efficiency
    print(f"Saving to {local_path}...")
    result_ds.to_parquet(local_path)

    # 5. Upload if requested
    if args.push_to_hub and args.output_repo:
        print(f"Uploading to HuggingFace Hub: {args.output_repo}...")
        try:
            # We can push the dataset directly
            result_ds.push_to_hub(args.output_repo, split="train")
            print("✅ Upload complete")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print(
                "Ensure you are logged in with 'huggingface-cli login' or have a valid token."
            )


if __name__ == "__main__":
    main()
