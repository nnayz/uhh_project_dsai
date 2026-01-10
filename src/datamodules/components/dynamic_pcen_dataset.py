from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

class PrototypeDynamicArrayDataset(Dataset):
    def __init__(self, path: Path, features: dict, train_param: dict):
        self.path = path
        self.features = features
        self.train_param = train_param
        self.samples_per_class = train_param.n_shot * 2 # TODO: Why 2?
        self.segment_length = train_param.segment_length
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass

    def _update_seglen(self):
        pass

    def select_negative(self):
        pass

    def select_positive(self):
        pass

    def select_segment(self):
        pass

    def build_meta(self):
        pass

    def update_meta(self):
        pass

    def remove_short_negative_duration(self):
        pass

    def get_class_durations(self):
        pass


    def get_all_csv_files(self):
        pass

    def get_glob_cls_name(self, file: Path) -> str:
        pass

    def get_df_pos(self, file: Path) -> pd.DataFrame:
        pass

    def get_cls_list(self):
        pass

    def get_time(self):
        pass

    def time2frame(self):
        pass

    def get_class2int(self):
        pass

def calculate_mean_std():
    pass