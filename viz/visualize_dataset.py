#!/usr/bin/env python
"""
Standalone script to generate dataset visualizations.

Usage:
    python scripts/visualize_dataset.py [--config-path CONF] [--exp-name NAME]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from viz.dataset_visualizer import (
    DatasetAnalyzer,
    visualize_dataset_overview,
    visualize_class_distribution,
    visualize_data_statistics,
    generate_dataset_report,
)


def main(cfg: DictConfig):
    """
    Main visualization pipeline.

    Args:
        cfg: Hydra configuration.
    """
    print("\n" + "=" * 80)
    print("DATASET VISUALIZATION PIPELINE")
    print("=" * 80)

    # Create output directory
    output_dir = Path(cfg.runtime.log_dir) / "dataset_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Data paths:")
    print(f"  Training: {cfg.path.train_dir}")
    print(f"  Validation: {cfg.path.eval_dir}")
    print(f"  Evaluation: {cfg.path.test_dir}")

    # Initialize analyzer
    analyzer = DatasetAnalyzer(cfg)

    print("\n[1/5] Loading dataset statistics...")
    analyzer.load_all_statistics()

    print("\n[2/5] Generating dataset overview visualization...")
    visualize_dataset_overview(analyzer, output_dir)

    print("\n[3/5] Generating class distribution visualization...")
    visualize_class_distribution(analyzer, output_dir)

    print("\n[4/5] Generating data statistics visualization...")
    visualize_data_statistics(analyzer, output_dir)

    print("\n[5/5] Generating dataset report...")
    report = generate_dataset_report(analyzer, output_dir)

    print("\n" + report)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataset visualizations"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="conf",
        help="Path to config directory (default: conf)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="dataset_analysis",
        help="Experiment name (default: dataset_analysis)",
    )

    args = parser.parse_args()

    # Initialize Hydra
    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(
            config_name="config",
            overrides=[f"exp_name={args.exp_name}"],
        )
        main(cfg)
