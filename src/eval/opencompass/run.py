import argparse
import os
import sys

from opencompass.cli.main import main


def _extract_custom_args():
    """Parse and remove custom args before opencompass sees sys.argv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_ckpt', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    custom_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if custom_args.model_ckpt is not None:
        os.environ['OC_MODEL_CKPT'] = custom_args.model_ckpt
    if custom_args.temperature is not None:
        os.environ['OC_TEMPERATURE'] = str(custom_args.temperature)
    if custom_args.threshold is not None:
        os.environ['OC_THRESHOLD'] = str(custom_args.threshold)


if __name__ == '__main__':
    _extract_custom_args()
    main()

