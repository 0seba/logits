import torch
import torch.nn.functional as F

# NOTE: This code assumes 'sm' (probabilities tensor) is already defined in your notebook context.
# If testing separately, uncomment the following line to generate a dummy 'sm':
# sm = torch.softmax(torch.randn(1000, dtype=torch.float64), dim=-1)

# Recalculate target log probabilities in high precision (float64)
target_probs = sm.double()
target_log_probs = target_probs.log()

# --- Method 1: Float16 ---
# Cast to float16 and then back to float64 for KL calculation
log_probs_f16 = target_log_probs.half()
log_probs_f16_dequant = log_probs_f16.double()

# --- Method 2: Uint16 Quantization ---
# 1. Determine Range
# Filter out -inf values to find the effective range of finite log probs
# We only care about the range of values that are actually representable
finite_mask = torch.isfinite(target_log_probs)
if finite_mask.any():
    min_log_prob = target_log_probs[finite_mask].min()
    max_log_prob = target_log_probs[finite_mask].max()
else:
    # Fallback if all are -inf (unlikely for softmax output)
    min_log_prob = torch.tensor(-100.0, dtype=torch.float64)
    max_log_prob = torch.tensor(0.0, dtype=torch.float64)

# 2. Quantize
# Map [min_log_prob, max_log_prob] -> [0, 65535]
# Any value below min_log_prob (like -inf) will be clamped to min_log_prob
scale = 65535.0 / (max_log_prob - min_log_prob)

# Clamp values to the finite range
clamped_log_probs = torch.clamp(target_log_probs, min=min_log_prob, max=max_log_prob)

# Quantize to uint16 (stored as float/int for simulation steps)
q_vals = ((clamped_log_probs - min_log_prob) * scale).round()
q_vals = torch.clamp(q_vals, 0, 65535)  # Ensure strictly in range

# 3. Dequantize
# Convert back to log probability space
log_probs_uint16_dequant = (q_vals / scale) + min_log_prob

# --- Comparison ---
# KL Divergence: KL(P || Q) = sum(p(x) * (log(p(x)) - log(q(x))))
# Pytorch kl_div(input, target) expects input = log_probs (Q), target = probs (P)
# We use reduction='sum' to get the total KL divergence over the distribution
kl_f16 = F.kl_div(log_probs_f16_dequant, target_probs, reduction="sum")
kl_uint16 = F.kl_div(log_probs_uint16_dequant, target_probs, reduction="sum")

print(f"KL Divergence (Float16): {kl_f16.item():.10f}")
print(f"KL Divergence (Uint16):  {kl_uint16.item():.10f}")
