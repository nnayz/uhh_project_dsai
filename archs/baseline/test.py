"""Test / evaluation script placeholder for the few-shot baseline."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    print(f"Testing (placeholder). Checkpoint: {args.checkpoint}")


if __name__ == '__main__':
    main()
