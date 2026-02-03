# Chat Template Fix for Logit Extraction

## Problem

The extraction was returning 0 results because the model's chat template doesn't contain the `{% generation %}` keyword needed to identify assistant tokens.

## Solution

Use the custom chat template file that includes generation markers.

## Usage

Add the `--chat-template` argument to your extraction command:

```bash
python extract_logits.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset your/dataset \
  --output your/output-repo \
  --chat-template qwen_chat_template.jinja \
  [other arguments...]
```

## Verify It Works

Before running the full extraction, verify the template works:

```bash
python extract_logits.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset your/dataset \
  --output your/output-repo \
  --chat-template qwen_chat_template.jinja \
  --verify-mask \
  --limit 1
```

This will show you:
- Total tokens in the conversation
- Number of assistant tokens detected
- The actual assistant text being extracted

You should see:
- ✅ "Assistant tokens: X" where X > 0 (not 0!)
- ✅ The decoded assistant text matches the expected response

## For Other Models

If you're using a different model (not Qwen), you may need to create a custom template for that model. The key requirement is to wrap the assistant's response in `{% generation %}...{% endgeneration %}` tags.

Example structure:
```jinja
{% for message in messages %}
  {% if message['role'] == 'user' %}
    {{ user_prefix }}{{ message['content'] }}{{ user_suffix }}
  {% elif message['role'] == 'assistant' %}
    {{ assistant_prefix }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ assistant_suffix }}
  {% endif %}
{% endfor %}
```
