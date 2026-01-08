import os
from glob import glob
from typing import Iterable

import pandas as pd


def _get_class_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    if "ME" in base:
        return "ME"
    if "BUK" in base:
        return "PB"
    return "HB"


def convert_single_file(file_path: str, save_path: str) -> None:
    def generate_line(class_name, start, end, filename):
        return "%s\t%s\t%s\t%s\n" % (class_name, start, end, filename)

    content = "event_label\tonset\toffset\tfilename\n"
    raw_result = pd.read_csv(file_path)

    for _, row in raw_result.iterrows():
        fname, start, end = row["Audiofilename"], row["Starttime"], row["Endtime"]
        class_name = "VAL@%s" % (_get_class_from_filename(fname))
        line = generate_line(
            class_name=class_name,
            start=start,
            end=end,
            filename=os.path.basename(fname).replace(".wav", ".csv"),
        )
        content = content + line

    with open(save_path, "w") as f:
        f.write(content)


def convert_eval_val(search_dirs: Iterable[str]) -> None:
    for root in search_dirs:
        for file in glob(os.path.join(root, "*/Eval_VAL_*.csv")):
            convert_single_file(
                file,
                os.path.join(os.path.dirname(file), "PSDS_" + os.path.basename(file)),
            )


def calculate_psds(search_dirs: Iterable[str], eval_meta_dir: str) -> float:
    from psds_eval import PSDSEval, plot_psd_roc, plot_per_class_psd_roc

    dtc_threshold = 0.5
    gtc_threshold = 0.5
    cttc_threshold = 0.3
    alpha_ct = 0.0
    alpha_st = 0.0
    max_efpr = 100

    ground_truth_csv = os.path.join(eval_meta_dir, "subset_gt.csv")
    metadata_csv = os.path.join(eval_meta_dir, "subset_meta.csv")

    gt_table = pd.read_csv(ground_truth_csv, sep="\t")
    meta_table = pd.read_csv(metadata_csv, sep="\t")
    psds_eval = PSDSEval(
        dtc_threshold,
        gtc_threshold,
        cttc_threshold,
        ground_truth=gt_table,
        metadata=meta_table,
    )
    for root in search_dirs:
        for file in glob(os.path.join(root, "*/PSDS_Eval_*.csv")):
            det_t = pd.read_csv(file, sep="\t")
            psds_eval.add_operating_point(det_t)
    psds = psds_eval.psds(alpha_ct, alpha_st, max_efpr)
    plot_psd_roc(psds, filename="roc.png")
    tpr_vs_fpr, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)
    plot_per_class_psd_roc(
        tpr_vs_fpr,
        psds_eval.class_names,
        title="Per-class TPR-vs-FPR PSDROC",
        xlabel="FPR",
        filename="per_class_1.png",
    )
    plot_per_class_psd_roc(
        tpr_vs_efpr,
        psds_eval.class_names,
        title="Per-class TPR-vs-eFPR PSDROC",
        xlabel="eFPR",
        filename="per_class_2.png",
    )
    return float(psds.value)
