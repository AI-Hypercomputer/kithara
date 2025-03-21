import jax 
import jax.numpy as jnp

def dpo_loss_fn(logits, ref_logits, mask, tokens, beta = 0.1):
    
    # Split into chosen and rejected (even/odd indices)
    chosen_logits = logits[:, ::2]  # [batch_size, seq_len/2, vocab_size]
    rejected_logits = logits[:, 1::2]  # [batch_size, seq_len/2, vocab_size]
    ref_chosen_logits = ref_logits[:, ::2]
    ref_rejected_logits = ref_logits[:, 1::2]

    chosen_mask = mask[:, ::2]  # [batch_size, seq_len/2]
    rejected_mask = mask[:, 1::2]  # [batch_size, seq_len/2]
    chosen_tokens = tokens[:, ::2]  # [batch_size, seq_len/2]
    rejected_tokens = tokens[:, 1::2]  # [batch_size, seq_len/2]

    # Calculate log probabilities
    log_softmax = lambda x: x - jax.scipy.special.logsumexp(
        x, axis=-1, keepdims=True
    )
    chosen_log_probs = log_softmax(chosen_logits)
    rejected_log_probs = log_softmax(rejected_logits)
    ref_chosen_log_probs = log_softmax(ref_chosen_logits)
    ref_rejected_log_probs = log_softmax(ref_rejected_logits)

    # Get log probs for the actual tokens using JAX's advanced indexing
    batch_indices = jnp.arange(chosen_log_probs.shape[0])[:, None]
    seq_indices = jnp.arange(chosen_log_probs.shape[1])[None, :]

    # Gather relevant token log probs - shape: [batch, seq]
    chosen_token_log_probs = chosen_log_probs[
        batch_indices, seq_indices, chosen_tokens
    ]
    rejected_token_log_probs = rejected_log_probs[
        batch_indices, seq_indices, rejected_tokens
    ]
    ref_chosen_token_log_probs = ref_chosen_log_probs[
        batch_indices, seq_indices, chosen_tokens
    ]
    ref_rejected_token_log_probs = ref_rejected_log_probs[
        batch_indices, seq_indices, rejected_tokens
    ]

    # Calculate importance weights (ρ in the DPO paper)
    chosen_imp_weights = chosen_token_log_probs - ref_chosen_token_log_probs
    rejected_imp_weights = (
        rejected_token_log_probs - ref_rejected_token_log_probs
    )

    # Apply mask and average over valid tokens
    chosen_avg_imp = jnp.sum(
        chosen_imp_weights * chosen_mask, axis=1
    ) / jnp.maximum(jnp.sum(chosen_mask, axis=1), 1.0)
    rejected_avg_imp = jnp.sum(
        rejected_imp_weights * rejected_mask, axis=1
    ) / jnp.maximum(jnp.sum(rejected_mask, axis=1), 1.0)

    # DPO loss calculation
    logits_diff = chosen_avg_imp - rejected_avg_imp
    loss = -jax.nn.log_sigmoid(beta * logits_diff)

    return jnp.mean(loss)
