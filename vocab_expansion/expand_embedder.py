import argparse

import jax.numpy as jnp
import jax.random

from MaxText import checkpointing
from flax.training import train_state
from orbax.checkpoint import args as ocp_args


def _trunc_normal(key, shape, std: float, dtype):
    """Sample truncated normal ([-2, 2] standard deviations), then scale by std."""
    x = jax.random.truncated_normal(key, lower=-2.0, upper=2.0, shape=shape, dtype=dtype)
    return x * std


def view_checkpoint(checkpoint_path: str):
    manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_path, True, False, 1
    )

    restored = manager.restore(0)
    print(type(restored))
    items = restored['items']
    print(items.keys())
    params = items['params']['params']
    print(params.keys())

    embedder = params['token_embedder']
    print(embedder.keys())
    embeddings = embedder['embedding']
    print(embeddings.shape)


def add_embeddings(checkpoint_path: str, num_tokens: int = 2048, out_step: int = 0):
    manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_path, True, False, 1
    )

    restored = manager.restore(0)
    print(f"Restored checkpoint type: {type(restored)}")
    items = restored['items']
    print(f"Items keys: {items.keys()}")
    
    # Extract the full state, not just params
    state = items
    params = state['params']['params']
    print(f"Params keys: {params.keys()}")

    embedder = params['token_embedder']
    print(f"Embedder keys: {embedder.keys()}")
    embeddings = embedder['embedding']
    old_vocab_size, dim = embeddings.shape
    print(f"Original embedding shape: {embeddings.shape}")

    # Generate new embeddings
    key = jax.random.PRNGKey(67)  # 67!!
    new_rows = _trunc_normal(key, (num_tokens, dim), 0.02, dtype=embeddings.dtype)
    new_embeddings = jnp.concatenate((embeddings, new_rows), axis=0)
    print(f"New embedding shape: {new_embeddings.shape}")

    # Update only the embeddings in the params
    params['token_embedder']['embedding'] = new_embeddings

    # Reconstruct the full state, preserving all original fields
    # Use the original state structure and just update the params
    updated_state = train_state.TrainState(
        step=state.get('step', out_step),  # Preserve original step if it exists
        apply_fn=state.get('apply_fn', None),
        params={"params": params},
        tx=state.get('tx', None),
        opt_state=state.get('opt_state', {})
    )

    # Save the updated checkpoint with force flag to ensure overwriting
    print(f"Saving updated checkpoint to step {out_step}")
    saved = checkpointing.save_checkpoint(manager, out_step, updated_state, force=True)
    
    if saved:
        print("Checkpoint save initiated successfully")
    else:
        print("Warning: Checkpoint save may have been skipped")
    
    # Wait for the save to complete
    print("Waiting for checkpoint save to complete...")
    manager.wait_until_finished()
    print("Checkpoint saved and verified!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add new token embeddings to a Gemma3 checkpoint")
    parser.add_argument("--checkpoint-path", required=True,
                        help="Path to converted checkpoint root. Expects 0 step where just converted.")
    parser.add_argument("--num-tokens", type=int, default=2048,
                        help="How many tokens to add for embedding table (default: 2048)")
    parser.add_argument("--out-step", type=int, default=0,
                        help="Step number to save the updated checkpoint (default: 0, overwrites original)")
    parser.add_argument("--view-only", action="store_true",
                        help="Only view the checkpoint without modifying it")
    args = parser.parse_args()

    if args.view_only:
        view_checkpoint(args.checkpoint_path)
    else:
        add_embeddings(args.checkpoint_path, args.num_tokens, args.out_step)