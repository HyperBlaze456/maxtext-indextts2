import argparse
import json
from pathlib import Path

from MaxText import checkpointing
from flax.training import train_state  # only to wrap for saving as TrainState again



def view_checkpoint(checkpoint_path: str):

    manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_path, True, False, 1
    )

    restored = manager.restore(0)
    print(type(restored))

    print(restored['items'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Minimum model overview")
    parser.add_argument("--checkpoint-path", required=True,
                        help="Path to converted checkpoint root. Expects 0 step where just converted.")
    args = parser.parse_args()
    view_checkpoint(args.checkpoint_path)