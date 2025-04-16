import jax 
import jax.numpy as jnp

def dpo_loss_fn(logits, ref_logits, mask, tokens, beta=0.1, label_smoothing=0.0):
    jax.debug.print("logits {}", logits[0, 0, :10])
    jax.debug.print("ref_logits {}", ref_logits[0, 0, :10])
    
    # Split into chosen and rejected (even/odd indices)
    chosen_logits = logits[::2]  # [batch_size/2, seq_len, vocab_size]
    rejected_logits = logits[1::2]  # [batch_size/2, seq_len, vocab_size]
    ref_chosen_logits = ref_logits[::2]
    ref_rejected_logits = ref_logits[1::2]
    
    chosen_mask = mask[::2, :]  # [batch_size/2, seq_len]
    rejected_mask = mask[1::2, :]  # [batch_size/2, seq_len]
    chosen_tokens = tokens[::2, :]  # [batch_size/2, seq_len]
    rejected_tokens = tokens[1::2, :]  # [batch_size/2, seq_len]
    
    # Calculate log probabilities
    log_softmax = lambda x: x - jax.scipy.special.logsumexp(
        x, axis=-1, keepdims=True
    )
    chosen_log_probs = log_softmax(chosen_logits)
    rejected_log_probs = log_softmax(rejected_logits)
    ref_chosen_log_probs = log_softmax(ref_chosen_logits)
    ref_rejected_log_probs = log_softmax(ref_rejected_logits)

    # Get log probs for the actual tokens using JAX's advanced indexing
    batch_size, seq_len = chosen_tokens.shape
    vocab_size = chosen_logits.shape[-1]
    
    # Create indices for gathering
    batch_indices = jnp.arange(batch_size)[:, None]
    seq_indices = jnp.arange(seq_len)[None, :]
    
    jax.debug.print("chosen_tokens {}", chosen_tokens.ravel()[:10])
    jax.debug.print("chosen_tokens shape {}", chosen_tokens.shape)

    chosen_token_logits= chosen_logits[
        batch_indices, seq_indices, chosen_tokens
    ]
    rejected_token_logits= rejected_logits[
        batch_indices, seq_indices, rejected_tokens
    ]

    jax.debug.print("chosen_token_logits {}", chosen_token_logits[0, :10])
    jax.debug.print("rejected_token_logits {}", rejected_token_logits[0, :10])
    
    
    # Gather relevant token log probs
    chosen_token_log_probs = chosen_log_probs[
        batch_indices, seq_indices, chosen_tokens
    ]
    rejected_token_log_probs = rejected_log_probs[
        batch_indices, seq_indices, rejected_tokens
    ]
    jax.debug.print("chosen_token_log_probs {}", chosen_token_logits[0, :10])
    jax.debug.print("rejected_token_log_probs {}", rejected_token_log_probs[0, :10])

    all_chosen_log_probs = jnp.sum(chosen_token_log_probs * chosen_mask, axis=-1)
    all_rejected_log_probs = jnp.sum(rejected_token_log_probs * rejected_mask, axis=-1)

    jax.debug.print("all_chosen_log_probs {}", all_chosen_log_probs)
    jax.debug.print("all_rejected_log_probs {}", all_rejected_log_probs)
    
    
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
    
    # Calculate average log probabilities for metrics
    # chosen_logps = jnp.sum(
    #     chosen_token_log_probs * chosen_mask, axis=1
    # ) / jnp.maximum(jnp.sum(chosen_mask, axis=1), 1.0)
    # rejected_logps = jnp.sum(
    #     rejected_token_log_probs * rejected_mask, axis=1
    # ) / jnp.maximum(jnp.sum(rejected_mask, axis=1), 1.0)
        
    # DPO loss calculation
    logits_diff = chosen_avg_imp - rejected_avg_imp
    # jax.debug.print("logits_diff: {}", logits_diff)
    
    # Apply label smoothing to the loss
    smooth_target = 1.0 - label_smoothing
    
    # The standard DPO loss is -log(sigmoid(beta * logits_diff))
    # With label smoothing, we interpolate between that and -log(sigmoid(-beta * logits_diff))
    positive_loss = -jax.nn.log_sigmoid(beta * logits_diff)
    negative_loss = -jax.nn.log_sigmoid(-beta * logits_diff)
    
    loss = smooth_target * positive_loss + label_smoothing * negative_loss
    
    # jax.debug.print("loss: {}", loss)
    
    # Calculate rewards (r_θ in the paper)
    # chosen_rewards = beta * chosen_avg_imp
    # rejected_rewards = beta * rejected_avg_imp
    
    # Calculate reward accuracies (how often chosen > rejected)
    # reward_accuracies = (chosen_rewards > rejected_rewards).astype(jnp.float32)


    mean_chosen_logits = jnp.sum(chosen_logits * chosen_mask ) / jnp.maximum(jnp.sum(chosen_mask), 1.0)
    mean_rejected_logits = jnp.sum(rejected_logits * rejected_mask ) / jnp.maximum(jnp.sum(rejected_mask), 1.0)


    # Return metrics dictionary along with the loss
    metrics = {
        # "rewards/chosen": jnp.mean(chosen_rewards),
        # "rewards/rejected": jnp.mean(rejected_rewards),
        # "rewards/accuracies": jnp.mean(reward_accuracies),
        # "rewards/margins": jnp.mean(chosen_rewards - rejected_rewards),
        "logps/chosen": jnp.mean(all_chosen_log_probs), #
        "logps/rejected": jnp.mean(all_rejected_log_probs),#
        "logits/chosen": mean_chosen_logits,#
        "logits/rejected": mean_rejected_logits,#
    }
    
    # Return both the loss and the metrics dictionary
    return jnp.mean(loss), metrics
