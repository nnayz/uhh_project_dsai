import pytorch_lightning as L
from torch import nn
from torch.nn import functional as F
import torch
from typing import Tuple, List, Any
import os
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
import pickle
from rich.progress import Progress


def save_pickle(obj: Any, fname: str | Path) -> None:
    """Save an object to a pickle file."""
    fname = Path(fname)
    print(f"Save pickle at {fname}")
    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(fname: str | Path) -> Any:
    """
    Load an object from a pickle file.
    """
    fname = Path(fname)
    print(f"Load pickle at {fname}")
    with open(fname, "rb") as f:
        return pickle.load(f)


class PrototypeModule(L.LightningModule):
    """
    lightning module organises the code into five sections:
    - Computations (init)
    - Train loop (training_step)
    - Validation loop (validation_step)
    - Test loop (test_step)
    - Optimisers (configure_optimizers)

    """

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model
        self.onset_offset: dict[str, dict[str, np.ndarray]] = {}
        self.fps: float = 1.0  # Will be set from hparams if available

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ) -> torch.Tensor:
        if self.hparams.train.negative_train_contrast:
            (x, x_neg, y, y_neg, class_name) = batch

            x = torch.cat([x, x_neg], dim=0)
            y = torch.cat([y, y_neg], dim=0)

        else:
            x, y, class_name = batch

        x_out = self.forward(x)

        if self.hparams.train.negative_train_contrast:
            from src.utils.loss import prototypical_loss_filter_negative as loss_fn
        else:
            from src.utils.loss import prototypical_loss as loss_fn

        tr_loss, tr_acc, tr_supcon = loss_fn(x_out, y, self.hparams.train.n_shot)
        return tr_loss, tr_acc, tr_supcon

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ) -> torch.Tensor:
        tr_loss, tr_acc, tr_supcon = self.step(batch, batch_idx)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_epoch=True, on_step=True)
        self.log("train/loss", tr_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", tr_acc, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": tr_loss + tr_supcon, "acc": tr_acc}

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ) -> torch.Tensor:
        val_loss, val_acc, val_supcon = self.step(batch, batch_idx)
        self.log("val/loss", val_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val/acc", val_acc, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": val_loss + val_supcon, "acc": val_acc}

    def on_test_epoch_start(self) -> None:
        """Initialize onset_offset dictionary at the start of test epoch."""
        self.onset_offset = {}
        # Set fps from hparams if available
        if hasattr(self, "hparams") and hasattr(self.hparams, "features"):
            if hasattr(self.hparams.features, "sr") and hasattr(
                self.hparams.features, "hop_mel"
            ):
                self.fps = self.hparams.features.sr / self.hparams.features.hop_mel
            elif hasattr(self.hparams.features, "fps"):
                self.fps = self.hparams.features.fps

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ) -> dict[str, Any]:
        def transform(
            a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Extract first element from batch dimension."""
            return a[0, ...], b[0, ...], c[0, ...]

        (
            (
                X_pos,
                X_neg,
                X_query,
                X_pos_neg,
                X_neg_neg,
                X_query_neg,
                hop_seg,
                hop_seg_neg,
                max_len,
                neg_min_length,
            ),
            strt_index_query,
            audio_name,
            seg_len,
        ) = batch
        X_pos, X_neg, X_query = transform(X_pos, X_neg, X_query)
        X_pos_neg, X_neg_neg, X_query_neg = transform(X_pos_neg, X_neg_neg, X_query_neg)
        hop_seg, hop_seg_neg, strt_index_query, max_len, neg_min_length, seg_len = (
            hop_seg[0, ...].item(),
            hop_seg_neg[0, ...].item(),
            strt_index_query[0, ...].item(),
            max_len[0, ...].item(),
            neg_min_length[0, ...].item(),
            seg_len[0, ...].item(),
        )
        # onset, offset = self.evaluate_prototypes(X_pos, X_neg, X_query, hop_seg, strt_index_query, mask)
        # onset_neg, offset_neg = self.evaluate_prototypes(X_pos_neg, X_neg_neg, X_query_neg, hop_seg_neg, strt_index_query, mask)
        padding_len = seg_len // 2
        onset_offset = self.evaluate_prototypes(
            X_pos, X_neg, X_query, hop_seg, strt_index_query, audio_name[0]
        )

        if self.hparams.train.negative_seg_search:
            onset_offset_neg = self.evaluate_prototypes(
                X_pos_neg,
                X_neg_neg,
                X_query_neg,
                hop_seg_neg,
                strt_index_query,
                audio_name[0],
            )

        audio_name = os.path.basename(audio_name[0])

        for k in onset_offset.keys():
            if k not in self.onset_offset.keys():
                self.onset_offset[k] = {}
                self.onset_offset[k]["name_arr"] = np.array([])
                self.onset_offset[k]["onset_arr"] = np.array([])
                self.onset_offset[k]["offset_arr"] = np.array([])

            if self.hparams.train.negative_seg_search:
                neg_onset_offset = []
                start = 0
                for on, off in zip(onset_offset_neg[k][0], onset_offset_neg[k][1]):
                    end = on
                    neg_onset_offset.append((start, end))
                    start = off

            # Use the detected negative samples to perform post-processing (splitting)
            if self.hparams.train.negative_seg_search:
                onset_offset[k][0], onset_offset[k][1] = self.splitting_segment(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    neg_onset_offset,
                    max_len,
                )

            # Remove long segment
            if self.hparams.train.remove_long_segment:
                onset_offset[k][0], onset_offset[k][1] = self.remove_long_segment(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    neg_min_length,
                    max_len,
                )

            # Use the detected negative samples to perform post-processing (merging)
            if self.hparams.train.merging_segment:
                onset_offset[k][0], onset_offset[k][1] = self.merging_segment(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    neg_min_length,
                    max_len,
                )

            # Remove long segment
            if self.hparams.train.remove_long_segment:  # TODO
                onset_offset[k][0], onset_offset[k][1] = self.remove_long_segment(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    neg_min_length,
                    max_len,
                )

            # Padding Tail
            if self.hparams.train.padd_tail:  # TODO
                onset_offset[k][0], onset_offset[k][1] = self.padding_tail(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    padding_len / self.fps,
                )

            name = np.repeat(audio_name, len(onset_offset[k][0]))

            self.onset_offset[k]["name_arr"] = np.append(
                self.onset_offset[k]["name_arr"], name
            )
            self.onset_offset[k]["onset_arr"] = np.append(
                self.onset_offset[k]["onset_arr"], onset_offset[k][0]
            )
            self.onset_offset[k]["offset_arr"] = np.append(
                self.onset_offset[k]["offset_arr"], onset_offset[k][1]
            )

        import time

        time.sleep(0.1)  # TODO: remove it if CPU resource is not bottleneck
        return {}

    def convert_single_file(self, file_path: str | Path, save_path: str | Path) -> None:
        """Convert evaluation file to PSDS format."""

        def get_class(fname: str) -> str:
            fname = Path(fname).name
            if "ME" in fname:
                return "ME"
            elif "BUK" in fname:
                return "PB"
            return "HB"

        def generate_a_line(
            class_name: str, start: float, end: float, filename: str
        ) -> str:
            return f"{class_name}\t{start}\t{end}\t{filename}\n"

        content = "event_label\tonset\toffset\tfilename\n"
        raw_result = pd.read_csv(file_path)

        for _, row in raw_result.iterrows():
            fname, start, end = row["Audiofilename"], row["Starttime"], row["Endtime"]
            class_name = f"VAL@{get_class(fname)}"
            filename = Path(fname).name.replace(".wav", ".csv")
            content += generate_a_line(
                class_name=class_name, start=start, end=end, filename=filename
            )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(content)

    def convert_eval_val(self) -> None:
        """Convert all evaluation VAL files to PSDS format."""
        for file in glob("*/Eval_VAL_*.csv"):
            file_path = Path(file)
            save_path = file_path.parent / f"PSDS_{file_path.name}"
            self.convert_single_file(file_path, save_path)

    def calculate_psds(self) -> None:
        """Calculate PSDS (Polyphonic Sound Detection Score) metrics."""
        try:
            from psds_eval import PSDSEval, plot_psd_roc, plot_per_class_psd_roc
        except ImportError:
            print("Warning: psds_eval not available, skipping PSDS calculation")
            return

        dtc_threshold = 0.5
        gtc_threshold = 0.5
        cttc_threshold = 0.3
        alpha_ct = 0.0
        alpha_st = 0.0
        max_efpr = 100

        # Try to get folder path from hparams or use default
        folder_path = (
            getattr(self.hparams.path, "eval_meta", Path("eval_meta"))
            if hasattr(self, "hparams")
            else Path("eval_meta")
        )
        ground_truth_csv = Path(folder_path) / "subset_gt.csv"
        metadata_csv = Path(folder_path) / "subset_meta.csv"

        if not ground_truth_csv.exists() or not metadata_csv.exists():
            print(f"Warning: Ground truth or metadata files not found at {folder_path}")
            return

        gt_table = pd.read_csv(ground_truth_csv, sep="\t")
        meta_table = pd.read_csv(metadata_csv, sep="\t")
        psds_eval = PSDSEval(
            dtc_threshold,
            gtc_threshold,
            cttc_threshold,
            ground_truth=gt_table,
            metadata=meta_table,
        )
        for file in glob("*/PSDS_Eval_*.csv"):
            det_t = pd.read_csv(file, sep="\t")
            psds_eval.add_operating_point(det_t)
        psds = psds_eval.psds(alpha_ct, alpha_st, max_efpr)
        print(f"\nPSDS-Score: {psds.value:.5f}")
        print("Saving pickle!")
        save_pickle(psds, "psds.pkl")
        plot_psd_roc(psds, filename="roc.png")
        tpr_vs_fpr, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)
        plot_per_class_psd_roc(
            tpr_vs_fpr,
            psds_eval.class_names,
            title="Per-class TPR-vs-FPR PSDROC",
            xlabel="FPR",
            filename="per_class_1.png",
        )
        save_pickle(tpr_vs_fpr, "tpr_vs_fpr.pkl")
        save_pickle(psds_eval.class_names, "class_names.pkl")
        plot_per_class_psd_roc(
            tpr_vs_efpr,
            psds_eval.class_names,
            title="Per-class TPR-vs-eFPR PSDROC",
            xlabel="eFPR",
            filename="per_class_2.png",
        )
        save_pickle(tpr_vs_efpr, "tpr_vs_efpr.pkl")
        self.log("psds", psds.value)

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch to process results."""
        # self.split_long_segments_based_on_energy() # TODO checkout if you need this function
        best_result = None
        best_f_measure = 0.0
        for k in self.onset_offset.keys():
            df_out = pd.DataFrame(
                {
                    "Audiofilename": [
                        Path(x).name for x in self.onset_offset[k]["name_arr"]
                    ],
                    "Starttime": self.onset_offset[k]["onset_arr"],
                    "Endtime": self.onset_offset[k]["offset_arr"],
                }
            )
            output_dir = Path(str(k))
            output_dir.mkdir(exist_ok=False)
            csv_path = output_dir / "Eval_raw.csv"
            df_out.to_csv(csv_path, index=False)
            # Postprocessing and evaluate
            res = self.post_process(alpha=k)
            res_new = self.post_process_new(alpha=k)

            if res["fmeasure"] > best_f_measure:
                best_result = res
                best_f_measure = res["fmeasure"]
            if res_new["fmeasure"] > best_f_measure:
                best_result = res_new
                best_f_measure = res_new["fmeasure"]
        print("Best Result: ")
        print(best_result)
        if best_result:
            for k, v in best_result.items():
                self.log(str(k), v)
        # New evaluation method
        self.convert_eval_val()
        self.calculate_psds()

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        pass

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optim = torch.optim.Adam(
            [{"params": self.model.parameters()}], lr=self.hparams.train.lr_rate
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim,
            gamma=self.hparams.train.scheduler_gamma,
            step_size=self.hparams.train.scheduler_step_size,
        )
        return [optim], [lr_scheduler]

    # def merging_segment(self, pos_onset_offset, neg_min_length, max_len):
    #     onset, offset = [], []
    #     i = 0
    #     while(i < len(pos_onset_offset)):

    #         if(i >= len(pos_onset_offset) - 1): return onset, offset

    #         on, off = pos_onset_offset[i]
    #         on_next, off_next = pos_onset_offset[i+1]
    #         # on_next_next, off_next_next = pos_onset_offset[i+2]

    #         # if(is_all_negative(on, off)): continue
    #         # Divide big segment into small segments
    #         if((off_next-on) * self.fps < 1.2*max_len or (on_next-off) * self.fps < 0.8 * neg_min_length):
    #             onset.append(on)
    #             offset.append(off_next)
    #             while((off_next-on) * self.fps < 1.2*max_len or (on_next-off) * self.fps < 0.8 * neg_min_length):
    #                 i += 1
    #                 offset[-1] = off_next
    #                 off = off_next
    #                 on_next, off_next = pos_onset_offset[i+1]
    #             print("merge", (off_next-on) * self.fps, (on_next-off) * self.fps, max_len, neg_min_length)

    #         # Do not change this segment
    #         else:
    #             onset.append(on)
    #             offset.append(off)
    #         i += 1

    #     return onset, offset

    def merging_segment(
        self,
        pos_onset_offset: List[Tuple[float, float]],
        neg_min_length: float,
        max_len: float,
    ) -> Tuple[List[float], List[float]]:
        """Merge segments that are close together."""
        onset, offset = [], []
        i = 0

        while i < len(pos_onset_offset):
            if i >= len(pos_onset_offset) - 1:
                on, off = pos_onset_offset[i]
                onset.append(on)
                offset.append(off)
                break

            on, off = pos_onset_offset[i]
            on_next, off_next = pos_onset_offset[i + 1]

            if (off_next - on) * self.fps < max_len or (
                on_next - off
            ) * self.fps < 0.5 * neg_min_length:
                onset.append(on)
                offset.append(off_next)
                while (off_next - on) * self.fps < max_len or (
                    on_next - off
                ) * self.fps < 0.5 * neg_min_length:
                    i += 1
                    if i >= len(pos_onset_offset) - 1:
                        break
                    offset[-1] = off_next
                    off = off_next
                    on_next, off_next = pos_onset_offset[i + 1]
                print(
                    f"merge: duration={(off_next - on) * self.fps:.2f}, "
                    f"gap={(on_next - off) * self.fps:.2f}, "
                    f"max_len={max_len}, neg_min_length={neg_min_length}"
                )
            else:
                onset.append(on)
                offset.append(off)
            i += 1

        return onset, offset

    def remove_long_segment(
        self,
        pos_onset_offset: List[Tuple[float, float]],
        neg_min_length: float,
        max_len: float,
    ) -> Tuple[List[float], List[float]]:
        """Remove segments that are too long."""
        onset, offset = [], []
        for on, off in pos_onset_offset:
            segment_length = (off - on) * self.fps
            if segment_length > 2 * max_len:
                print(
                    f"-remove: length={segment_length:.2f}, "
                    f"max_len={max_len}, ratio={segment_length / (2 * max_len):.2f}"
                )
            else:
                onset.append(on)
                offset.append(off)
        return onset, offset

    def padding_tail(
        self, pos_onset_offset: List[Tuple[float, float]], padding_len: float
    ) -> Tuple[List[float], List[float]]:
        """Add padding to segments that have sufficient spacing."""
        if len(pos_onset_offset) <= 1:
            return [on for on, _ in pos_onset_offset], [
                off for _, off in pos_onset_offset
            ]

        onset, offset = [], []
        for i in range(len(pos_onset_offset)):
            if i == 0:
                on, off = pos_onset_offset[i]
                onset.append(on)
                offset.append(off)
                continue
            if i >= len(pos_onset_offset) - 1:
                on, off = pos_onset_offset[i]
                onset.append(on)
                offset.append(off)
                break

            prev_on, prev_off = pos_onset_offset[i - 1]
            on, off = pos_onset_offset[i]
            next_on, next_off = pos_onset_offset[i + 1]

            if (
                next_on - off > 0.1 + 2 * padding_len
                and on - prev_off > 0.1 + 2 * padding_len
            ):
                onset.append(on - padding_len)
                offset.append(off + padding_len)
                print(
                    f"++padding: on={on:.2f}, off={off:.2f}, padding={padding_len:.2f}"
                )
            else:
                onset.append(on)
                offset.append(off)
        return onset, offset

    def splitting_segment(
        self,
        pos_onset_offset: List[Tuple[float, float]],
        neg_onset_offset: List[Tuple[float, float]],
        max_len: float,
    ) -> Tuple[List[float], List[float]]:
        """Split long segments based on negative segment boundaries."""

        def split_segment(
            _onset: float, _offset: float
        ) -> Tuple[List[float], List[float]]:
            """Divide the predicted positive segment into smaller ones based on negative segment estimation."""
            ret_onset, ret_offset = [], []
            breakpoints = [
                (neg_on, neg_off)
                for neg_on, neg_off in neg_onset_offset
                if _onset < neg_on < _offset and neg_off < _offset
            ]

            _on = _onset
            for break_on, break_off in breakpoints:
                _off = break_on
                ret_onset.append(_on)
                ret_offset.append(_off)
                _on = break_off
            # Add the last segment
            if _on < _offset:
                ret_onset.append(_on)
                ret_offset.append(_offset)
            return ret_onset, ret_offset

        onset, offset = [], []
        for on, off in pos_onset_offset:
            segment_length = (off - on) * self.fps
            if segment_length > max_len * 2:
                print(
                    f"> splitting: length={segment_length:.2f}, "
                    f"threshold={max_len * 2}, ratio={segment_length / (max_len * 2):.2f}"
                )
                new_onset, new_offset = split_segment(on, off)
                onset.extend(new_onset)
                offset.extend(new_offset)
            else:
                onset.append(on)
                offset.append(off)
        return onset, offset

    def split_long_segments_based_on_energy(self) -> None:
        """Split long segments based on energy analysis (requires librosa and scipy)."""
        try:
            import librosa
            from scipy.signal import find_peaks, peak_widths
        except ImportError:
            print(
                "Warning: librosa and scipy required for split_long_segments_based_on_energy"
            )
            return

        print("Splitting long segments!!!!!!!")
        name_arr_temp, onset_arr_temp, offset_arr_temp = [], [], []
        for i in range(self.name_arr.shape[0]):
            name, max_len, onset, offset = (
                self.name_arr[i],
                self.max_len_arr[i],
                self.onset_arr[i],
                self.offset_arr[i],
            )
            # Split long segment based on their energy
            if "ML_126376" in name and (offset - onset) * self.fps > 2 * max_len:
                print((offset - onset) * self.fps / max_len)
                x, sr = librosa.load(
                    name, offset=onset, duration=offset - onset, sr=None
                )
                rms = librosa.feature.rms(y=x)
                rms = rms / np.max(rms)
                rms = rms[0, ...]
                peaks, _ = find_peaks(rms)
                _, weight, start, end = peak_widths(rms, peaks, rel_height=0.5)

                start *= 512 / sr
                end *= 512 / sr
                start += onset
                end += onset

                threshold = np.max(weight) / 2
                start = start[weight > threshold]
                end = end[weight > threshold]
                start[0] = onset
                end[-1] = offset
                for s, e in zip(start, end):
                    name_arr_temp.append(name)
                    onset_arr_temp.append(s)
                    offset_arr_temp.append(e)
            else:
                name_arr_temp.append(name)
                onset_arr_temp.append(onset)
                offset_arr_temp.append(offset)
        self.name_arr, self.onset_arr, self.offset_arr = (
            np.array(name_arr_temp),
            np.array(onset_arr_temp),
            np.array(offset_arr_temp),
        )

    def log_result(
        self,
        overall_scores: dict[str, Any],
        scores_per_set: dict[str, dict[str, Any]],
        scores_per_audiofile: dict[str, dict[str, Any]],
        name: str,
    ) -> None:
        """Log results at different granularities."""
        self.log_final_result(overall_scores, name)
        self.log_result_for_each_set(scores_per_set, name)
        self.log_result_for_each_audio_file(scores_per_audiofile, name)

    def log_final_result(
        self, overall_scores: dict[str, Any], name: str = "test"
    ) -> None:
        """Log overall scores."""
        for k, v in overall_scores.items():
            value = v.item() if hasattr(v, "item") else v
            self.log(f"{name}-overall_scores/{k}", value)

    def log_result_for_each_set(
        self, scores_per_set: dict[str, dict[str, Any]], name: str = "test"
    ) -> None:
        """Log scores for each dataset."""
        cache = {}
        for dataset, scores in scores_per_set.items():
            for k, v in scores.items():
                cache[f"{k}/{dataset}"] = v
        for k, v in cache.items():
            self.log(f"{name}-each_set_scores/{k}", v)

    def log_result_for_each_audio_file(
        self, scores_per_audio_file: dict[str, dict[str, Any]], name: str = "test"
    ) -> None:
        """Log scores for each audio file."""
        cache = {}
        for audiofile, scores in scores_per_audio_file.items():
            filename = Path(audiofile).name
            for k, v in scores.items():
                cache[f"{k}/{filename}"] = v
        for k, v in cache.items():
            self.log(f"{name}-each_audiofile/{k}", v)

    def post_process_test(self, dataset="TEST", alpha=0.9):
        from src.utils.evaluation import evaluate
        from src.utils.post_proc import post_processing

        test_path = self.hparams.path.test_dir
        if test_path[-1] != "/":
            test_path += "/"

        evaluation_file = "%s/Eval_raw.csv" % alpha
        save_path = "%s" % alpha

        best_result = None
        for threshold in np.arange(0.1, 0.9, 0.1):
            print("Threshold %s" % threshold)
            team_name = "Baseline" + str(threshold)
            new_evaluation_file = "%s/Eval_%s_threshold_ada_postproc_%s.csv" % (
                alpha,
                dataset,
                threshold,
            )
            post_processing(
                test_path, evaluation_file, new_evaluation_file, threshold=threshold
            )

    def post_process_new_test(self, dataset="TEST", alpha=0.9):
        from src.utils.evaluation import evaluate
        from src.utils.post_proc_new import post_processing

        test_path = self.hparams.path.test_dir
        if test_path[-1] != "/":
            test_path += "/"

        evaluation_file = "%s/Eval_raw.csv" % alpha
        save_path = "%s" % alpha

        best_result = None
        for threshold_length in np.arange(0.05, 0.25, 0.05):
            team_name = "Baseline" + str(threshold_length)
            print("Threshold length %s" % threshold_length)
            new_evaluation_file = "%s/Eval_%s_threshold_fix_length_postproc_%s.csv" % (
                alpha,
                dataset,
                threshold_length,
            )
            post_processing(
                test_path,
                evaluation_file,
                new_evaluation_file,
                threshold_length=threshold_length,
            )

    def post_process(self, dataset: str = "VAL", alpha: float = 0.9) -> dict[str, Any]:
        """Post-process evaluation results with adaptive threshold."""
        from src.utils.evaluation import evaluate
        from src.utils.post_proc import post_processing

        val_path = Path(self.hparams.path.eval_dir)
        val_path_str = (
            str(val_path) + "/" if not str(val_path).endswith("/") else str(val_path)
        )

        evaluation_file = Path(f"{alpha}/Eval_raw.csv")
        save_path = str(alpha)

        print("Before preprocessing: ")
        team_name = "Baseline_unprocessed"

        (
            overall_scores,
            individual_file_result,
            scores_per_set,
            scores_per_audiofile,
        ) = evaluate(str(evaluation_file), val_path_str, team_name, dataset, save_path)

        self.log_result(
            overall_scores,
            scores_per_set=scores_per_set,
            scores_per_audiofile=scores_per_audiofile,
            name="No_Post",
        )

        best_result = None
        for threshold in np.arange(0.2, 0.6, 0.1):
            print(f"Threshold {threshold}")
            team_name = f"Baseline{threshold}"
            new_evaluation_file = Path(
                f"{alpha}/Eval_{dataset}_threshold_ada_postproc_{threshold}.csv"
            )
            post_processing(
                val_path_str,
                str(evaluation_file),
                str(new_evaluation_file),
                threshold=threshold,
            )
            (
                overall_scores,
                individual_file_result,
                scores_per_set,
                scores_per_audiofile,
            ) = evaluate(
                str(new_evaluation_file), val_path_str, team_name, dataset, save_path
            )
            if (
                best_result is None
                or best_result[0]["fmeasure"] < overall_scores["fmeasure"]
            ):
                best_result = (
                    overall_scores,
                    individual_file_result,
                    threshold,
                    scores_per_set,
                    scores_per_audiofile,
                )
        print("******************BEST RESULT*****************")
        if best_result:
            for k, v in best_result[1].items():
                print(f"{k}: {v}")
            print(f"Scores: {best_result[0]}, Threshold: {best_result[2]}")
            self.log_result(
                best_result[0],
                best_result[3],
                best_result[4],
                name=f"proc_thresh_minlen_{best_result[2]:.2f}",
            )
            return best_result[0]
        return {}

    def post_process_new(
        self, dataset: str = "VAL", alpha: float = 0.9
    ) -> dict[str, Any]:
        """Post-process evaluation results with fixed length threshold."""
        from src.utils.evaluation import evaluate
        from src.utils.post_proc_new import post_processing

        val_path = Path(self.hparams.path.eval_dir)
        val_path_str = (
            str(val_path) + "/" if not str(val_path).endswith("/") else str(val_path)
        )

        evaluation_file = Path(f"{alpha}/Eval_raw.csv")
        save_path = str(alpha)

        best_result = None
        for threshold_length in np.arange(0.05, 0.25, 0.05):
            team_name = f"Baseline{threshold_length}"
            print(f"Threshold length {threshold_length}")
            new_evaluation_file = Path(
                f"{alpha}/Eval_{dataset}_threshold_fix_length_postproc_{threshold_length}.csv"
            )
            post_processing(
                val_path_str,
                str(evaluation_file),
                str(new_evaluation_file),
                threshold_length=threshold_length,
            )
            (
                overall_scores,
                individual_file_result,
                scores_per_set,
                scores_per_audiofile,
            ) = evaluate(
                str(new_evaluation_file), val_path_str, team_name, dataset, save_path
            )
            if (
                best_result is None
                or best_result[0]["fmeasure"] < overall_scores["fmeasure"]
            ):
                best_result = (
                    overall_scores,
                    individual_file_result,
                    threshold_length,
                    scores_per_set,
                    scores_per_audiofile,
                )
        print("******************BEST RESULT*****************")
        if best_result:
            for k, v in best_result[1].items():
                print(f"{k}: {v}")
            print(f"Scores: {best_result[0]}, Threshold: {best_result[2]}")
            self.log_result(
                best_result[0],
                best_result[3],
                best_result[4],
                name=f"proc_by_length_{best_result[2]:.2f}",
            )
            return best_result[0]
        return {}

    def get_probability(
        self,
        proto_pos: torch.Tensor,
        neg_proto: torch.Tensor,
        query_set_out: torch.Tensor,
    ) -> List[float]:
        """Calculate the probability of each query point belonging to the positive class.

        Args:
            proto_pos: Positive class prototype
            neg_proto: Negative class prototype calculated from randomly chosen segments
            query_set_out: Model output for query set samples

        Returns:
            Probability array for the positive class
        """
        prototypes = torch.stack([proto_pos, neg_proto]).squeeze(1)
        dists = self.euclidean_dist(query_set_out, prototypes)
        logits = -dists
        prob = torch.softmax(logits, dim=1)
        prob_pos = prob[:, 0]
        return prob_pos.detach().cpu().toList()

    def get_probability_old(
        self, x_pos: torch.Tensor, neg_proto: torch.Tensor, query_set_out: torch.Tensor
    ) -> List[float]:
        """Calculate probability using old method (computes prototype from x_pos).

        Args:
            x_pos: Model output for the positive class
            neg_proto: Negative class prototype
            query_set_out: Model output for query set samples

        Returns:
            Probability array for the positive class
        """
        pos_prototype = x_pos.mean(0)
        prototypes = torch.stack([pos_prototype, neg_proto])
        dists = self.euclidean_dist(query_set_out, prototypes)
        inverse_dist = torch.div(1.0, dists)
        prob = torch.softmax(inverse_dist, dim=1)
        prob_pos = prob[:, 0]
        return prob_pos.detach().cpu().toList()

    # def concate_mask(self, x, mask):
    #     import ipdb; ipdb.set_trace()
    #     return x

    def concate_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Concatenate mask with input tensor along channel dimension.

        Args:
            x: Input tensor of shape [batch, time, features]
            mask: Mask tensor of shape [batch, mask_features]

        Returns:
            Concatenated tensor
        """
        pad_length = x.size(2) - mask.size(1)
        mask = F.pad(mask, (0, pad_length))
        mask = mask.unsqueeze(1).expand(x.size(0), x.size(1), -1)
        return torch.cat([x.unsqueeze(1), mask.unsqueeze(1)], dim=1)

    def evaluate_prototypes(
        self, X_pos, X_neg, X_query, hop_seg, strt_index_query=None, audio_name=None
    ):
        X_pos = torch.tensor(X_pos)
        Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
        X_neg = torch.tensor(X_neg)
        Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
        X_query = torch.tensor(X_query)
        Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

        # num_batch_query = len(Y_query) // self.hparams.eval.query_batch_size
        query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
        q_loader = torch.utils.data.DataLoader(
            dataset=query_dataset,
            batch_sampler=None,
            batch_size=self.hparams.eval.query_batch_size,
            shuffle=False,
        )
        # query_set_feat = torch.zeros(0, 48).cpu()
        # batch_samplr_pos = EpisodicBatchSampler(Y_pos, 2, 1, self.hparams.train.n_shot)
        pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
        pos_loader = torch.utils.data.DataLoader(
            dataset=pos_dataset, batch_sampler=None
        )
        "List for storing the combined probability across all iterations"
        prob_comb = []
        emb_dim = self.hparams.features.embedding_dim
        pos_set_feat = torch.zeros(0, emb_dim).cpu()

        print("Creating positive prototype")
        device = next(self.model.parameters()).device

        with Progress() as progress:
            pos_task = progress.add_task(
                "[green]Processing positive samples", total=len(pos_loader)
            )
            for batch in pos_loader:
                x, y = batch
                feat = self.model(x.to(device))
                feat = feat.cpu()
                feat_mean = feat.mean(dim=0).unsqueeze(0)
                pos_set_feat = torch.cat((pos_set_feat, feat_mean), dim=0)
                progress.update(pos_task, advance=1)
        pos_proto = pos_set_feat.mean(dim=0)

        iterations = self.hparams.eval.iterations
        for i in range(iterations):
            prob_pos_iter = []
            neg_indices = torch.randperm(len(X_neg))[: self.hparams.eval.samples_neg]
            X_neg_ind = X_neg[neg_indices]
            Y_neg_ind = Y_neg[neg_indices]
            feat_neg = self.model(X_neg_ind.to(device))
            feat_neg = feat_neg.detach().cpu()
            proto_neg = feat_neg.mean(dim=0)
            q_iterator = iter(q_loader)

            print(f"Iteration number {i}")
            with Progress() as progress:
                query_task = progress.add_task(
                    f"[cyan]Iteration {i+1}/{iterations}", total=len(q_loader)
                )
                for batch in q_iterator:
                    x_q, y_q = batch
                    x_query = self.model(x_q.to(device))

                    proto_neg = proto_neg.detach().cpu()
                    x_query = x_query.detach().cpu()

                    probability_pos = self.get_probability(
                        pos_proto, proto_neg, x_query
                    )
                    prob_pos_iter.extend(probability_pos)
                    progress.update(query_task, advance=1)

            prob_comb.append(prob_pos_iter)

        prob_final = np.mean(np.array(prob_comb), axis=0)
        # Save the probability here to perform model ensemble
        filename = Path(audio_name).stem
        prob_dir = Path("prob_comb")
        prob_dir.mkdir(exist_ok=True)
        np.save(prob_dir / f"{filename}.npy", np.array(prob_comb))

        thresh_List = np.arange(0, 1, 0.05)
        onset_offset_ret = {}
        hop_mel = self.hparams.features.hop_mel
        sr = self.hparams.features.sr

        str_time_query = strt_index_query * hop_mel / sr

        for thresh in thresh_List:
            krn = np.array([1, -1])
            prob_thresh = np.where(prob_final > thresh, 1, 0)
            changes = np.convolve(krn, prob_thresh)

            onset_frames = np.where(changes == 1)[0]
            offset_frames = np.where(changes == -1)[0]

            onset = onset_frames * hop_seg * hop_mel / sr + str_time_query
            offset = offset_frames * hop_seg * hop_mel / sr + str_time_query

            assert len(onset) == len(offset)
            onset_offset_ret[thresh] = [onset, offset]

        # from scipy.signal import medfilt

        # # Use median filtering
        # for thresh in thresh_List:
        #     krn = np.array([1, -1])
        #     prob_thresh = np.where(medfilt(prob_final) > thresh, 1, 0)
        #     # prob_pos_final = prob_final * prob_thresh

        #     changes = np.convolve(krn, prob_thresh)

        #     onset_frames = np.where(changes == 1)[0]
        #     offset_frames = np.where(changes == -1)[0]

        #     str_time_query = (
        #         strt_index_query * self.hparams.features.hop_mel / self.hparams.features.sr
        #     )

        #     onset = (
        #         (onset_frames)
        #         * (hop_seg)
        #         * self.hparams.features.hop_mel
        #         / self.hparams.features.sr
        #     )
        #     onset = onset + str_time_query

        #     offset = (
        #         (offset_frames)
        #         * (hop_seg)
        #         * self.hparams.features.hop_mel
        #         / self.hparams.features.sr
        #     )
        #     offset = offset + str_time_query

        #     assert len(onset) == len(offset)

        #     # Use median filtering (plus 1)
        #     onset_offset_ret[thresh+1] = [onset, offset]

        return onset_offset_ret

    def euclidean_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute euclidean distance between two tensors.

        Args:
            x: Tensor of shape [N, D]
            y: Tensor of shape [M, D]

        Returns:
            Distance matrix of shape [N, M]
        """
        if x.size(1) != y.size(1):
            raise ValueError(
                f"Feature dimensions must match: {x.size(1)} != {y.size(1)}"
            )

        n, m, d = x.size(0), y.size(0), x.size(1)
        x_expanded = x.unsqueeze(1).expand(n, m, d)
        y_expanded = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x_expanded - y_expanded, 2).sum(2)

    def cosine_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cosine distance between two tensors.

        Args:
            x: Tensor of shape [N, D]
            y: Tensor of shape [M, D]

        Returns:
            Distance matrix of shape [N, M]
        """
        if x.size(1) != y.size(1):
            raise ValueError(
                f"Feature dimensions must match: {x.size(1)} != {y.size(1)}"
            )

        n, m, d = x.size(0), y.size(0), x.size(1)
        x_expanded = x.unsqueeze(1).expand(n, m, d)
        y_expanded = y.unsqueeze(0).expand(n, m, d)
        return -torch.nn.CosineSimilarity(dim=2, eps=1e-6)(x_expanded, y_expanded)


if __name__ == "__main__":

    def calculate_psds():
        from glob import glob
        from psds_eval import PSDSEval, plot_psd_roc, plot_per_class_psd_roc

        dtc_threshold = 0.5
        gtc_threshold = 0.5
        cttc_threshold = 0.3
        alpha_ct = 0.0
        alpha_st = 0.0
        max_efpr = 100
        ground_truth_csv = Path(
            "/vol/research/dcase2022/project/hhlab/src/models/eval_meta/subset_gt.csv"
        )
        metadata_csv = Path(
            "/vol/research/dcase2022/project/hhlab/src/models/eval_meta/subset_meta.csv"
        )
        gt_table = pd.read_csv(ground_truth_csv, sep="\t")
        meta_table = pd.read_csv(metadata_csv, sep="\t")
        psds_eval = PSDSEval(
            dtc_threshold,
            gtc_threshold,
            cttc_threshold,
            ground_truth=gt_table,
            metadata=meta_table,
        )
        for file in glob(
            "/vol/research/dcase2022/project/t5_open_source/DCASE_2022_Task_5/logs/experiments/runs/final/2022-07-05_19-25-40/*/PSDS_Eval_*.csv"
        ):
            det_t = pd.read_csv(file, sep="\t")
            psds_eval.add_operating_point(det_t)
            break
        psds = psds_eval.psds(alpha_ct, alpha_st, max_efpr)
        print(f"\nPSDS-Score: {psds.value:.5f}")
        print("Saving pickle!")
        save_pickle(psds, "psds.pkl")
        plot_psd_roc(psds, filename="roc.png")
        tpr_vs_fpr, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)
        plot_per_class_psd_roc(
            tpr_vs_fpr,
            psds_eval.class_names,
            title="Per-class TPR-vs-FPR PSDROC",
            xlabel="FPR",
            filename="per_class_1.png",
        )
        save_pickle(tpr_vs_fpr, "tpr_vs_fpr.pkl")
        save_pickle(psds_eval.class_names, "class_names.pkl")
        plot_per_class_psd_roc(
            tpr_vs_efpr,
            psds_eval.class_names,
            title="Per-class TPR-vs-eFPR PSDROC",
            xlabel="eFPR",
            filename="per_class_2.png",
        )
        save_pickle(tpr_vs_efpr, "tpr_vs_efpr.pkl")

    calculate_psds()
