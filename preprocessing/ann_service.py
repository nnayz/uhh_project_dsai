from pathlib import Path
from typing import List, Union, Tuple
from itertools import chain
import pandas as pd

def parse_positive_events_train(
    csv_path: Union[str, Path],
    glob_cls_name: str,
    padding: float = 0.025,
) -> List[Tuple[float, float, str]]:
    """
    Parse POS events for the training set with legacy logic preserved.

    - If a "CALL" column exists, use the folder-derived class name.
    - Otherwise, use the column names that contain POS labels.
    """
    df = pd.read_csv(csv_path, header=0, index_col=False)
    df_pos = df[(df == "POS").any(axis=1)].copy()
    df_pos.loc[:, "Starttime"] = df_pos["Starttime"] - padding
    df_pos.loc[:, "Endtime"] = df_pos["Endtime"] + padding

    start_time = [start for start in df_pos["Starttime"]]
    end_time = [end for end in df_pos["Endtime"]]

    if "CALL" in df_pos.columns:
        cls_list = [glob_cls_name] * len(start_time)
    else:
        cls_list = [
            df_pos.columns[(df_pos == "POS").loc[index]].values
            for index, _ in df_pos.iterrows()
        ]
        cls_list = list(chain.from_iterable(cls_list))

    return list(zip(start_time, end_time, cls_list))


def parse_positive_events_val(
    csv_path: Union[str, Path],
    glob_cls_name: str,
    padding: float = 0.025,
) -> List[Tuple[float, float, str]]:
    """
    Parse POS events for the validation set with legacy logic preserved.

    Expects the Q column at index 3 (HB format) and uses CSV stem as class name.
    """
    df = pd.read_csv(csv_path, header=0, index_col=False)
    df_pos = df[(df == "POS").any(axis=1)].copy()
    df_pos.loc[:, "Starttime"] = df_pos["Starttime"] - padding
    df_pos.loc[:, "Endtime"] = df_pos["Endtime"] + padding

    start_time = [start for start in df_pos["Starttime"]]
    end_time = [end for end in df_pos["Endtime"]]

    if "Q" == df_pos.columns[3]:
        cls_list = [glob_cls_name] * len(start_time)
    else:
        raise ValueError(
            "Unsupported validation CSV format: expected Q column at index 3."
        )

    return list(zip(start_time, end_time, cls_list))
