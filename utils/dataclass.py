import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path
import random
from omegaconf import DictConfig


class PrototypicalDataset(Dataset):
    """
    Generic dataset class for Prototypical Networks.

    This dataset organizes data by classes and supports episodic sampling
    for few-shot learning tasks. Each episode consists of:
    - Support set: K examples per class (used to compute prototypes)
    - Query set: Examples to classify (used for training/evaluation)
    """

    def __init__(
        self,
        data: Dict[str, List],  # {class_name: [examples]}
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        mode: str = "train",
        seed: Optional[int] = None,
    ):
        """
        Initializes the PrototypicalDataset.
        Args:
            data: Dictionary mapping class names to lists of examples
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            mode: "train", "val", or "evaluation"
            seed: Random seed for reproducibility
        """
        self.data = data
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.mode = mode
        self.seed = seed

        # Get all class names
        self.classes = list(data.keys())

        # Validate that we have enough classes
        if len(self.classes) < n_way:
            raise ValueError(
                f"Not enough classes: have {len(self.classes)}, need {n_way}"
            )

        # Validate that each class has enough examples
        min_examples = k_shot + n_query
        for class_name, examples in data.items():
            if len(examples) < min_examples:
                raise ValueError(
                    f"Class {class_name} has only {len(examples)} examples, "
                    f"need at least {min_examples} (k_shot={k_shot}, n_query={n_query})"
                )

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __len__(self) -> int:
        """
        Returns the number of episodes that can be created.
        For training, we can create many episodes by sampling different classes.
        """
        # Return a large number for training (episodes are sampled randomly)
        # For validation/evaluation, return a fixed number of episodes
        if self.mode == "train":
            return 10000  # Arbitrary large number
        else:
            # For val/evaluation, create episodes from all possible class combinations
            return min(1000, len(self.classes) // self.n_way)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single episode consisting of support and query sets.

        Returns:
            support_x: Support set features [n_way * k_shot, ...]
            support_y: Support set labels [n_way * k_shot]
            query_x: Query set features [n_way * n_query, ...]
            query_y: Query set labels [n_way * n_query]
        """
        # Sample N classes randomly
        selected_classes = random.sample(self.classes, self.n_way)

        support_x = []
        support_y = []
        query_x = []
        query_y = []

        for class_idx, class_name in enumerate(selected_classes):
            examples = self.data[class_name]

            # Randomly sample k_shot + n_query examples
            sampled_indices = random.sample(
                range(len(examples)), self.k_shot + self.n_query
            )

            # Split into support and query
            support_indices = sampled_indices[: self.k_shot]
            query_indices = sampled_indices[self.k_shot :]

            # Add support examples
            for idx in support_indices:
                example = examples[idx]
                # Convert to tensor if not already
                if isinstance(example, np.ndarray):
                    example = torch.from_numpy(example).float()
                elif not isinstance(example, torch.Tensor):
                    example = torch.tensor(example, dtype=torch.float32)

                support_x.append(example)
                support_y.append(class_idx)

            # Add query examples
            for idx in query_indices:
                example = examples[idx]
                # Convert to tensor if not already
                if isinstance(example, np.ndarray):
                    example = torch.from_numpy(example).float()
                elif not isinstance(example, torch.Tensor):
                    example = torch.tensor(example, dtype=torch.float32)

                query_x.append(example)
                query_y.append(class_idx)

        # Stack into tensors
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y, dtype=torch.long)

        return support_x, support_y, query_x, query_y


class DataClass:
    """
    Main data class for loading and organizing data for Prototypical Networks.

    This class handles:
    - Loading data from directories
    - Organizing data by classes
    - Creating datasets for training/validation/evaluation
    - Creating DataLoaders for episodic training
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Hydra DictConfig containing data paths and parameters
        """
        self.cfg = cfg
        self.data_by_class: Dict[str, List] = {}
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

    def load_data(
        self,
        data_dir: Path,
        split: str = "train",
        class_dirs: Optional[List[str]] = None,
    ) -> Dict[str, List]:
        """
        Loads data from directory structure organized by classes.

        Args:
            data_dir: Path to directory containing class subdirectories
            split: "train", "val", or "evaluation"
            class_dirs: Optional list of class directory names to load.
                       If None, loads all subdirectories.

        Returns:
            Dictionary mapping class names to lists of examples
        """
        data_dir = Path(data_dir)

        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        data_by_class = {}

        # Get class directories
        if class_dirs is None:
            class_dirs = [
                d.name
                for d in data_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

        for class_dir_name in class_dirs:
            class_path = data_dir / class_dir_name

            if not class_path.exists() or not class_path.is_dir():
                continue

            # Load all examples for this class
            examples = self._load_class_examples(class_path)

            if len(examples) > 0:
                data_by_class[class_dir_name] = examples

        self.data_by_class = data_by_class
        self.class_to_idx = {name: idx for idx, name in enumerate(data_by_class.keys())}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        return data_by_class

    def _load_class_examples(self, class_path: Path) -> List:
        """
        Loads all examples from a class directory.

        This is a generic method that should be overridden for specific data types.
        For now, it returns a placeholder structure.

        Args:
            class_path: Path to class directory

        Returns:
            List of examples (should be tensors/arrays)
        """
        examples = []

        # Find all data files (common audio/image extensions)
        extensions = [".wav", ".mp3", ".flac", ".npy", ".pt", ".pth", ".jpg", ".png"]

        for ext in extensions:
            for file_path in class_path.glob(f"*{ext}"):
                # Load the file based on extension
                if ext == ".npy":
                    example = np.load(file_path)
                    examples.append(example)
                elif ext in [".pt", ".pth"]:
                    example = torch.load(file_path)
                    examples.append(example)
                # For other formats, you would add specific loaders
                # For now, we'll assume preprocessing has been done

        return examples

    def create_dataset(
        self,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        mode: str = "train",
        seed: Optional[int] = None,
    ) -> PrototypicalDataset:
        """
        Creates a PrototypicalDataset from loaded data.

        Args:
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            mode: "train", "val", or "evaluation"
            seed: Random seed for reproducibility

        Returns:
            PrototypicalDataset instance
        """
        if not self.data_by_class:
            raise ValueError("No data loaded. Call load_data() first.")

        return PrototypicalDataset(
            data=self.data_by_class,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            mode=mode,
            seed=seed,
        )

    def create_dataloader(
        self,
        dataset: PrototypicalDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Creates a DataLoader for episodic training.

        Note: batch_size should typically be 1 for episodic training,
        as each episode already contains multiple examples.

        Args:
            dataset: PrototypicalDataset instance
            batch_size: Batch size (usually 1 for episodes)
            shuffle: Whether to shuffle episodes
            num_workers: Number of worker processes

        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_class_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Returns class name to index and index to class name mappings.

        Returns:
            Tuple of (class_to_idx, idx_to_class) dictionaries
        """
        return self.class_to_idx, self.idx_to_class
