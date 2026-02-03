# Logit Extraction and Alignment Analysis

This repository contains tools for extracting token probabilities (logits) from Large Language Models and analyzing them aligned with the original conservation text.

## Overview

The core functionality consists of two main parts:
1.  **Extraction (`extract_logits.py`)**: Efficiently runs a model on a dataset, extracts the top-K and sampled logits for assistant responses, and saves them in a highly compressed format.
2.  **Analysis (`explore_logits.py`)**: Loads the compressed logit data, reconstructs the original conversation, aligned tokens, and visualizes the model's confidence.

**Supported Use Cases:**
*   **Models**: Instruction fine-tuned models (e.g., Chat/Instruct versions). Base models that do not support chat templates are not supported.
*   **Datasets**: Single-turn chat datasets (User query -> Assistant response). Multi-turn conversations are not currently supported.

## Data Structure

The extracted data is saved in a compressed format to minimize storage size while retaining precision where it matters.

### Storage Format
The data is stored in a Hugging Face dataset with the following schema:

*   **`top_logits_quantized`** (`bytes`): Top-K log probabilities quantized to `uint8`.
*   **`top_indices_low`** (`list[uint16]`): Lower 16 bits of the Token IDs for the top-K logits.
*   **`top_indices_high`** (`bytes`): Upper bits of the Token IDs, packed into 2-bit chunks (4 indices per byte).
*   **`sampled_logits_quantized`** (`bytes`): Log probabilities for the sampled nucleus, quantized to `uint8`.
*   **`sampled_indices_low`** / **`sampled_indices_high`**: Same packed format for the sampled indices.
*   **`logsumexp`** (`float`): The LogSumExp of the logits, used to reconstruct exact probabilities.
*   **`num_tokens`** (`int`): Total number of extracted tokens in the sample.

### Index Unpacking
Token IDs are split to save space, as most tokens fit in 16 bits, but vocabularies > 65536 require more. The unpacking logic is:

```python
full_index = low_bits | (high_bits << 16)
```

The `high_bits` are extracted from the packed bytes where each byte holds 4 high-bit entries:
- Bits 0-1: Index 0
- Bits 2-3: Index 1
- Bits 4-5: Index 2
- Bits 6-7: Index 3

## Prerequisites: Chat Templates & Assistant Masks

⚠️ **CRITICAL**: This tool relies on the tokenizer's chat template producing an `assistant_mask` to identify which tokens belong to the model's output.
Many default chat templates (e.g., standard Llama 2/3 templates from Hugging Face) **DO NOT** produce this mask by default.

If your model's default template does not support `assistant_masks` (or if you get 0 logit tokens extracted), you **MUST** provide a custom chat template that defines the generation structure.

## Usage

### 1. Extraction

Run `extract_logits.py` to process a dataset. Only data falling under the assistant mask will be extracted.

```bash
python extract_logits.py \
    --model_id "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_id "MegaScience/MegaScience" \
    --output_repo "user/my-logits-dataset" \
    --chat_template "path/to/custom_template.jinja" 
```

**Note:** You can pass a path to a jinja file or the template string itself to `--chat_template`.

### 2. Analysis & Visualization

Use `explore_logits.py` or the provided notebook to verify and explore the data. If the extraction used a custom template, you should likely use the same one here for correct alignment.

```python
from explore_logits import execute_analysis

# You can pass the template as a string or file path
my_template = "{% for message in messages %}...{% endfor %}"

execute_analysis(
    logit_dataset_id="user/my-logits-dataset",
    sample_index=0,
    chat_template=my_template
)
```

**Alignment Logic:**
The analysis script (`analyze_token_probabilities`) performs the following steps to align extracted logits with the text:
1.  **Reconstruction**: Re-applies the chat template (provided or default) to the original dataset sample to regenerate the exact input tokens seen by the model.
2.  **Masking**: Replicates the filtering logic used during extraction (excluding think tags, end tokens, etc.) to identify exactly which tokens were skipped.
3.  **Matching**: Maps the remaining "valid" tokens to the extracted logit sequence 1:1.
4.  **Prediction**: For each real token, it retrieves the top predicted tokens from the logits to compare what the model *wanted* to say vs what was actually said.

### 3. Decompression

The `explore_logits.py` script provides helper functions to handle the data:

- `unpack_indices(low, high)`: Reconstructs full 32-bit token IDs.
- `dequantize_all_logits(sample, index)`: Retrieves the full probability distribution (Nucleus + Sampled) for a specific token position.

## Visualization Output

The analysis tool produces a detailed line-by-line view:

```text
Detailed Token Analysis:
================================================================================
Real Token                     | Extracted Predictions (Top 5)
--------------------------------------------------------------------------------
"The"                          | "The"(0.85), "A"(0.10), ...
"extraction"                   | "extraction"(0.99), ...
```

- **Real Token**: The actual token from the dataset.
- **Predictions**: The model's top predicted next tokens with their probabilities.
- **Skipped**: Tokens marked as skipped (e.g., `<think>`) will be greyed out.
