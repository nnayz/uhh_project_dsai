"""
Dataset visualization module for understanding dataset composition and statistics.

This module provides comprehensive visualizations for:
- Dataset distribution across subsets (training, validation, evaluation)
- Class distribution across different folders
- Recording duration statistics
- Event density and frequency
- Sampling rate variations
- Number of events per class
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import seaborn as sns
from omegaconf import DictConfig
import warnings

warnings.filterwarnings("ignore")


class DatasetAnalyzer:
    """Analyzer for bioacoustic dataset statistics."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the analyzer.

        Args:
            cfg: Hydra config containing path information.
        """
        self.cfg = cfg
        self.train_dir = Path(cfg.path.train_dir)
        self.val_dir = Path(cfg.path.eval_dir)
        self.test_dir = Path(cfg.path.test_dir)

        self.train_stats = {}
        self.val_stats = {}
        self.test_stats = {}

    def _parse_csv_file(self, csv_path: Path) -> Dict:
        """
        Parse annotation CSV file.

        Args:
            csv_path: Path to the CSV annotation file.

        Returns:
            Dictionary with class information.
        """
        try:
            df = pd.read_csv(csv_path, sep="\t")
            if df.empty:
                return {}

            classes = {}
            for col in df.columns[3:]:
                col_name = col.strip()
                pos_count = (df[col] == "POS").sum()
                neg_count = (df[col] == "NEG").sum()
                unk_count = (df[col] == "UNK").sum()

                if pos_count > 0:  # Only include if there's at least one positive
                    classes[col_name] = {
                        "pos": pos_count,
                        "neg": neg_count,
                        "unk": unk_count,
                    }

            return classes
        except Exception as e:
            print(f"Error parsing {csv_path}: {e}")
            return {}

    def analyze_directory(self, directory: Path, subset_name: str) -> Dict:
        """
        Analyze all audio files in a directory.

        Args:
            directory: Path to the directory containing audio files.
            subset_name: Name of the subset (e.g., 'training', 'validation').

        Returns:
            Dictionary with statistics for the subset.
        """
        stats = {
            "subset": subset_name,
            "folders": {},
            "total_recordings": 0,
            "total_classes": set(),
            "total_events": 0,
        }

        # Find all subdirectories
        subdirs = [d for d in directory.iterdir() if d.is_dir()]

        for subdir in subdirs:
            folder_name = subdir.name
            audio_files = list(subdir.glob("*.wav"))

            if not audio_files:
                continue

            folder_stats = {
                "name": folder_name,
                "num_recordings": len(audio_files),
                "classes": defaultdict(lambda: {"pos": 0, "neg": 0, "unk": 0}),
                "total_events": 0,
                "durations": [],
                "sampling_rates": [],
            }

            for audio_file in audio_files:
                csv_file = audio_file.with_suffix(".csv")

                if csv_file.exists():
                    classes = self._parse_csv_file(csv_file)
                    for class_name, counts in classes.items():
                        folder_stats["classes"][class_name]["pos"] += counts[
                            "pos"
                        ]
                        folder_stats["classes"][class_name]["neg"] += counts[
                            "neg"
                        ]
                        folder_stats["classes"][class_name]["unk"] += counts[
                            "unk"
                        ]
                        folder_stats["total_events"] += counts["pos"]
                        stats["total_classes"].add(class_name)

                # Try to get duration (this is approximate)
                folder_stats["durations"].append(0)  # Placeholder

            stats["folders"][folder_name] = folder_stats
            stats["total_recordings"] += folder_stats["num_recordings"]
            stats["total_events"] += folder_stats["total_events"]

        return stats

    def load_all_statistics(self) -> None:
        """Load statistics from all dataset subsets."""
        print(f"Analyzing training set at {self.train_dir}...")
        if self.train_dir.exists():
            self.train_stats = self.analyze_directory(
                self.train_dir, "Training"
            )

        print(f"Analyzing validation set at {self.val_dir}...")
        if self.val_dir.exists():
            self.val_stats = self.analyze_directory(
                self.val_dir, "Validation"
            )

        print(f"Analyzing evaluation set at {self.test_dir}...")
        if self.test_dir.exists():
            self.test_stats = self.analyze_directory(
                self.test_dir, "Evaluation"
            )


def visualize_dataset_overview(analyzer: DatasetAnalyzer, output_dir: Path = None) -> None:
    """
    Create comprehensive overview visualization of dataset.

    Args:
        analyzer: DatasetAnalyzer instance with loaded statistics.
        output_dir: Optional directory to save the figure.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Dataset composition (recordings)
    ax1 = fig.add_subplot(gs[0, 0])
    subsets = []
    recordings = []
    for stats in [analyzer.train_stats, analyzer.val_stats, analyzer.test_stats]:
        if stats:
            subsets.append(stats["subset"])
            recordings.append(stats["total_recordings"])

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    bars1 = ax1.bar(subsets, recordings, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_ylabel("Number of Recordings", fontsize=11, fontweight="bold")
    ax1.set_title("Total Recordings per Subset", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Total classes
    ax2 = fig.add_subplot(gs[0, 1])
    classes = []
    for stats in [analyzer.train_stats, analyzer.val_stats, analyzer.test_stats]:
        if stats:
            classes.append(len(stats["total_classes"]))

    bars2 = ax2.bar(subsets, classes, color=colors, alpha=0.8, edgecolor="black")
    ax2.set_ylabel("Number of Classes", fontsize=11, fontweight="bold")
    ax2.set_title("Total Classes (excl. UNK) per Subset", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Total events
    ax3 = fig.add_subplot(gs[1, 0])
    events = []
    for stats in [analyzer.train_stats, analyzer.val_stats, analyzer.test_stats]:
        if stats:
            events.append(stats["total_events"])

    bars3 = ax3.bar(subsets, events, color=colors, alpha=0.8, edgecolor="black")
    ax3.set_ylabel("Number of Events", fontsize=11, fontweight="bold")
    ax3.set_title("Total Events (POS) per Subset", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Folders in training set
    ax4 = fig.add_subplot(gs[1, 1])
    if analyzer.train_stats:
        folder_names = []
        folder_recordings = []
        for folder_name, folder_data in analyzer.train_stats["folders"].items():
            folder_names.append(folder_name)
            folder_recordings.append(folder_data["num_recordings"])

        bars4 = ax4.barh(
            folder_names,
            folder_recordings,
            color=plt.cm.Set3(np.linspace(0, 1, len(folder_names))),
            alpha=0.8,
            edgecolor="black",
        )
        ax4.set_xlabel("Number of Recordings", fontsize=11, fontweight="bold")
        ax4.set_title("Training Set - Recordings per Folder", fontsize=12, fontweight="bold")
        ax4.grid(axis="x", alpha=0.3)
        for i, bar in enumerate(bars4):
            width = bar.get_width()
            ax4.text(
                width,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(width)}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=10,
            )

    # 5. Events per folder in training set
    ax5 = fig.add_subplot(gs[2, :])
    if analyzer.train_stats:
        folder_names = []
        folder_events = []
        for folder_name, folder_data in analyzer.train_stats["folders"].items():
            folder_names.append(folder_name)
            folder_events.append(folder_data["total_events"])

        bars5 = ax5.bar(
            folder_names,
            folder_events,
            color=plt.cm.Set3(np.linspace(0, 1, len(folder_names))),
            alpha=0.8,
            edgecolor="black",
        )
        ax5.set_ylabel("Number of Events", fontsize=11, fontweight="bold")
        ax5.set_title("Training Set - Total Events per Folder", fontsize=12, fontweight="bold")
        ax5.grid(axis="y", alpha=0.3)
        for bar in bars5:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    fig.suptitle(
        "Dataset Overview - Training, Validation, and Evaluation Sets",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "dataset_overview.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'dataset_overview.png'}")
    plt.show()


def visualize_class_distribution(analyzer: DatasetAnalyzer, output_dir: Path = None) -> None:
    """
    Visualize class distribution across dataset.

    Args:
        analyzer: DatasetAnalyzer instance with loaded statistics.
        output_dir: Optional directory to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Class Distribution Analysis",
        fontsize=14,
        fontweight="bold",
    )

    # Training set - classes per folder
    ax = axes[0, 0]
    if analyzer.train_stats:
        folder_names = []
        class_counts = []
        for folder_name, folder_data in analyzer.train_stats["folders"].items():
            folder_names.append(folder_name)
            class_counts.append(len(folder_data["classes"]))

        bars = ax.bar(
            folder_names,
            class_counts,
            color=plt.cm.Set3(np.linspace(0, 1, len(folder_names))),
            alpha=0.8,
            edgecolor="black",
        )
        ax.set_ylabel("Number of Classes", fontsize=11, fontweight="bold")
        ax.set_title("Training Set - Classes per Folder", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # Top classes in training set
    ax = axes[0, 1]
    if analyzer.train_stats:
        class_events = defaultdict(int)
        for folder_data in analyzer.train_stats["folders"].values():
            for class_name, counts in folder_data["classes"].items():
                class_events[class_name] += counts["pos"]

        # Sort and get top 15
        top_classes = sorted(
            class_events.items(), key=lambda x: x[1], reverse=True
        )[:15]
        class_names = [c[0] for c in top_classes]
        class_event_counts = [c[1] for c in top_classes]

        bars = ax.barh(
            class_names,
            class_event_counts,
            color=plt.cm.Spectral(np.linspace(0, 1, len(class_names))),
            alpha=0.8,
            edgecolor="black",
        )
        ax.set_xlabel("Number of Events (POS)", fontsize=11, fontweight="bold")
        ax.set_title("Training Set - Top 15 Classes by Event Count", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    # Validation set info
    ax = axes[1, 0]
    if analyzer.val_stats:
        folder_names = []
        class_counts = []
        event_counts = []
        for folder_name, folder_data in analyzer.val_stats["folders"].items():
            folder_names.append(folder_name)
            class_counts.append(len(folder_data["classes"]))
            event_counts.append(folder_data["total_events"])

        x_pos = np.arange(len(folder_names))
        width = 0.35

        bars1 = ax.bar(
            x_pos - width / 2,
            class_counts,
            width,
            label="Classes",
            alpha=0.8,
            edgecolor="black",
        )
        ax2 = ax.twinx()
        bars2 = ax2.bar(
            x_pos + width / 2,
            event_counts,
            width,
            label="Events",
            alpha=0.8,
            color="orange",
            edgecolor="black",
        )

        ax.set_xlabel("Folder", fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of Classes", fontsize=11, fontweight="bold", color="C0")
        ax2.set_ylabel("Number of Events", fontsize=11, fontweight="bold", color="orange")
        ax.set_title("Validation Set - Classes and Events per Folder", fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(folder_names)
        ax.tick_params(axis="y", labelcolor="C0")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax.grid(axis="y", alpha=0.3)

    # Evaluation set info
    ax = axes[1, 1]
    if analyzer.test_stats:
        folder_names = []
        class_counts = []
        event_counts = []
        for folder_name, folder_data in analyzer.test_stats["folders"].items():
            folder_names.append(folder_name)
            class_counts.append(len(folder_data["classes"]))
            event_counts.append(folder_data["total_events"])

        x_pos = np.arange(len(folder_names))
        width = 0.35

        bars1 = ax.bar(
            x_pos - width / 2,
            class_counts,
            width,
            label="Classes",
            alpha=0.8,
            edgecolor="black",
        )
        ax2 = ax.twinx()
        bars2 = ax2.bar(
            x_pos + width / 2,
            event_counts,
            width,
            label="Events",
            alpha=0.8,
            color="orange",
            edgecolor="black",
        )

        ax.set_xlabel("Folder", fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of Classes", fontsize=11, fontweight="bold", color="C0")
        ax2.set_ylabel("Number of Events", fontsize=11, fontweight="bold", color="orange")
        ax.set_title("Evaluation Set - Classes and Events per Folder", fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(folder_names)
        ax.tick_params(axis="y", labelcolor="C0")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'class_distribution.png'}")
    plt.show()


def visualize_data_statistics(analyzer: DatasetAnalyzer, output_dir: Path = None) -> None:
    """
    Create statistics comparison visualization.

    Args:
        analyzer: DatasetAnalyzer instance with loaded statistics.
        output_dir: Optional directory to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Dataset Statistics Comparison",
        fontsize=14,
        fontweight="bold",
    )

    # Events per recording
    ax = axes[0, 0]
    subset_names = []
    events_per_recording = []
    for stats in [analyzer.train_stats, analyzer.val_stats, analyzer.test_stats]:
        if stats and stats["total_recordings"] > 0:
            subset_names.append(stats["subset"])
            events_per_recording.append(
                stats["total_events"] / stats["total_recordings"]
            )

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    bars = ax.bar(subset_names, events_per_recording, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Average Events per Recording", fontsize=11, fontweight="bold")
    ax.set_title("Event Density Across Subsets", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Classes per recording
    ax = axes[0, 1]
    subset_names = []
    classes_per_recording = []
    for stats in [analyzer.train_stats, analyzer.val_stats, analyzer.test_stats]:
        if stats and stats["total_recordings"] > 0:
            subset_names.append(stats["subset"])
            classes_per_recording.append(
                len(stats["total_classes"]) / stats["total_recordings"]
            )

    bars = ax.bar(subset_names, classes_per_recording, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Average Classes per Recording", fontsize=11, fontweight="bold")
    ax.set_title("Class Diversity Across Subsets", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Data distribution pie chart
    ax = axes[1, 0]
    subset_recordings = []
    for stats in [analyzer.train_stats, analyzer.val_stats, analyzer.test_stats]:
        if stats:
            subset_recordings.append(stats["total_recordings"])

    labels = [s["subset"] for s in [analyzer.train_stats, analyzer.val_stats, analyzer.test_stats] if s]
    colors_pie = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    wedges, texts, autotexts = ax.pie(
        subset_recordings,
        labels=labels,
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    ax.set_title("Recording Distribution Across Subsets", fontsize=12, fontweight="bold")

    # Event distribution pie chart
    ax = axes[1, 1]
    subset_events = []
    for stats in [analyzer.train_stats, analyzer.val_stats, analyzer.test_stats]:
        if stats:
            subset_events.append(stats["total_events"])

    wedges, texts, autotexts = ax.pie(
        subset_events,
        labels=labels,
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    ax.set_title("Event Distribution Across Subsets", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "data_statistics.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'data_statistics.png'}")
    plt.show()


def generate_dataset_report(analyzer: DatasetAnalyzer, output_dir: Path = None) -> str:
    """
    Generate a text report of dataset statistics.

    Args:
        analyzer: DatasetAnalyzer instance with loaded statistics.
        output_dir: Optional directory to save the report.

    Returns:
        String containing the report.
    """
    report = []
    report.append("=" * 80)
    report.append("DATASET STATISTICS REPORT")
    report.append("=" * 80)

    for stats in [analyzer.train_stats, analyzer.val_stats, analyzer.test_stats]:
        if not stats:
            continue

        report.append(f"\n### {stats['subset'].upper()} SET ###")
        report.append(f"Total Recordings: {stats['total_recordings']}")
        report.append(f"Total Classes (excl. UNK): {len(stats['total_classes'])}")
        report.append(f"Total Events (POS annotations): {stats['total_events']}")

        if stats["total_recordings"] > 0:
            report.append(
                f"Average Events per Recording: {stats['total_events'] / stats['total_recordings']:.2f}"
            )

        report.append(f"\nFolders in {stats['subset']}:")
        for folder_name, folder_data in stats["folders"].items():
            report.append(f"\n  {folder_name}:")
            report.append(f"    - Recordings: {folder_data['num_recordings']}")
            report.append(f"    - Classes: {len(folder_data['classes'])}")
            report.append(f"    - Total Events (POS): {folder_data['total_events']}")
            if folder_data["num_recordings"] > 0:
                report.append(
                    f"    - Events per Recording: {folder_data['total_events'] / folder_data['num_recordings']:.2f}"
                )

    report.append("\n" + "=" * 80)
    report_text = "\n".join(report)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "dataset_report.txt", "w") as f:
            f.write(report_text)
        print(f"Saved: {output_dir / 'dataset_report.txt'}")

    return report_text


if __name__ == "__main__":
    print("This module is meant to be imported and used with a Hydra config.")
    print("Example usage in a script:")
    print(
        """
    from hydra import initialize, compose
    from pathlib import Path
    from viz.dataset_visualizer import DatasetAnalyzer, visualize_dataset_overview
    
    with initialize(config_path="conf"):
        cfg = compose(config_name="config")
        analyzer = DatasetAnalyzer(cfg)
        analyzer.load_all_statistics()
        
        output_dir = Path("outputs/dataset_visualizations")
        visualize_dataset_overview(analyzer, output_dir)
    """
    )
