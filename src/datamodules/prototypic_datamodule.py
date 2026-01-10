import pytorch_lightning as L
from torch.utils.data import DataLoader, Dataset


class PrototypicDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self):
        pass

    def init(self):
        pass

    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass

