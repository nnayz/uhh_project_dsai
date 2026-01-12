# Audio Preprocessing and CSV-Driven Segmentation

This document explains how audio files are prepared for training and evaluation using their corresponding CSV annotation files. It follows the current, active code paths used by the training pipeline.

## Overview

The preprocessing pipeline is split into two layers:

1) **Offline feature export**: convert each `.wav` into cached feature arrays (`*_logmel.npy`, `*_pcen.npy`).
2) **CSV-driven segmentation**: read annotation CSVs, derive positive/negative regions, and slice cached features into fixed-length segments for training/validation/test.

The CSV-driven segmentation happens inside the `preprocessing/sequence_data/` datasets and is used by the DataModule.

## 1) Feature Export (Offline)

Entry point: `g5 export-features` → `preprocessing/feature_export.py:export_features()`

Steps:

1. **Collect audio paths**
   - `collect_wav_paths_from_dir()` recursively finds `.wav` under the configured split directory.
   - Paths come from `cfg.path.train_dir`, `cfg.path.eval_dir`, `cfg.path.test_dir`.

2. **Load waveform**
   - `preprocessing/preprocess.py:load_audio()` loads `.wav` with librosa.
   - Normalizes waveform to `[-1, 1]` if non-empty.

3. **Compute features**
   - `waveform_to_logmel()` or `waveform_to_pcen()` computes features based on `cfg.features`.
   - Supported suffixes: `logmel`, `pcen`.

4. **Save arrays**
   - For each suffix, write `audio_name_<suffix>.npy` next to the `.wav`.

These cached arrays are required for all dataset classes in `preprocessing/sequence_data/`.

## 2) Shared Feature Loading

Entry point: `preprocessing/sequence_data/pcen.py:Feature_Extractor`

Steps:

1. **Find `.wav` files**
   - `Feature_Extractor.__init__()` scans the provided audio roots to find `.wav` files.

2. **Compute normalization statistics**
   - `update_mean_std()` loads up to 1000 cached feature arrays per suffix.
   - Mean/std are stored in a class-level cache and reused.

3. **Load features per file**
   - `extract_feature(audio_path)` loads `*_suffix.npy` for each suffix in `cfg.features.feature_types`.
   - Normalizes `(array - mean) / std`.
   - Converts to time-major if needed (`_ensure_time_major()` expects `[frames, bins]`).

These steps are used by all dataset classes to read cached features before segmentation.

## 3) Training Dataset: CSV-Driven Segment Construction

Entry point: `preprocessing/sequence_data/dynamic_pcen_dataset.py:PrototypeDynamicArrayDataSet`

This is the default training dataset when `train_param.use_validation_first_5` is false.

### CSV parsing and metadata

1. **Collect CSVs**
   - `get_all_csv_files()` walks `cfg.path.train_dir` and returns all `*.csv` paths.

2. **Filter positives**
   - `get_df_pos()` reads a CSV and keeps rows where any column equals `POS`.

3. **Add 25 ms padding**
   - `get_time()` subtracts `0.025` from `Starttime` and adds `0.025` to `Endtime`.

4. **Infer class labels**
   - `get_cls_list()`
     - If a `CALL` column exists, uses the folder name as class.
     - Otherwise, uses columns containing `POS` as class labels.

5. **Build metadata**
   - `update_meta()` creates per-class structures:
     - `info`: positive segments `(start, end)`
     - `neg_info`: negative segments between positives
     - `duration`: positive duration
     - `file`: `.wav` path
   - Also loads the feature array into `self.pcen[audio_path]` using `Feature_Extractor.extract_feature()`.

### Segment sampling

1. **Select class**
   - `__getitem__()` maps the incoming index to a class name.

2. **Select positive segment**
   - `select_positive()` chooses a random `(start, end)` from `meta[class].info`.

3. **Select negative segment (optional)**
   - If `train_param.negative_train_contrast` is true, `select_negative()` samples from `neg_info` and enforces minimum length.

4. **Crop or tile to fixed length**
   - `select_segment()` converts `(start, end)` to frame indices and slices the cached feature array.
   - If the segment is too short, it tiles to `seg_len` frames.

5. **Return tensors**
   - Returns `segment, label, class_name` or `(pos, neg, pos_label, neg_label, class_name)`.

## 4) Training Dataset with Eval Bootstrapping

Entry point: `preprocessing/sequence_data/dynamic_pcen_dataset_first_5.py:PrototypeDynamicArrayDataSetWithEval`

Used when `train_param.use_validation_first_5` is true.

Key differences from the default dataset:

1. **CSV sources**
   - Combines train CSVs, eval CSVs, and optionally extra train CSVs (`path.extra_train_dir`).

2. **Eval CSV sampling**
   - `get_df_pos()` uses only the first `n_shot` annotations from eval CSVs.

3. **Class buckets**
   - Tracks `train_classes`, `eval_classes`, and `extra_train_classes` separately.

4. **Negative duration filtering**
   - `remove_short_negative_duration()` drops classes where negative segments are too short.

5. **Segment buffering for extra classes**
   - Maintains `segment_buffer` and `start_end_buffer` to stabilize extra class sampling within a batch.

All other steps (meta building, segment slicing) follow the same pattern as the default dataset.

## 5) Validation Dataset

Entry point: `preprocessing/sequence_data/dynamic_pcen_dataset_val.py:PrototypeDynamicArrayDataSetVal`

Steps:

1. **Collect eval CSVs**
   - `get_all_csv_files()` walks `cfg.path.eval_dir` for `*.csv`.

2. **Infer classes**
   - `get_glob_cls_name()` uses the CSV filename as class name.
   - `get_cls_list()` uses `Q` (the label column) to map to a single class.

3. **Build metadata**
   - Similar to training, but negative segments are derived as gaps between positives in the same file.

4. **Sparse negative fallback**
   - If total negative duration < 2.0 seconds, it treats the remainder of the file as negative.

5. **Build feature buffers**
   - `build_buffer()` caches normalized features per audio file.

The dataset returns segments in the same format as training, and provides `eval_class_idxs` for the sampler.

## 6) Test Dataset (Fixed Segment Length)

Entry point: `preprocessing/sequence_data/test_loader.py:PrototypeTestSet`

Steps:

1. **Collect eval CSVs**
   - `get_all_csv_files()` walks `cfg.path.eval_dir` for `*.csv`.

2. **Identify label column**
   - `find_positive_label()` finds the column containing `Q`.

3. **Convert times to frames**
   - `time_2_frame()` pads by 25 ms and converts start/end to frame indices.

4. **Create support (positive) set**
   - Uses the first `n_shot` positive events as support examples.

5. **Create negative set**
   - Slides across the file with a hop of `eval_param.hop_seg`.

6. **Create query set**
   - Starts after the last support event, slides with the same hop.

7. **Return test batch**
   - Returns `(X_pos, X_neg, X_query, hop_seg)` plus `strt_index_query` and `audio_path`.

## 7) Test Dataset (Adaptive Segment Length)

Entry point: `preprocessing/sequence_data/test_loader_ada_seglen_better_neg_v2.py:PrototypeAdaSeglenBetterNegTestSetV2`

Steps (high level, following the code flow):

1. **Collect eval/test CSVs**
   - Walks `cfg.path.eval_dir` and `cfg.path.test_dir` for `*.csv`.

2. **Identify label column**
   - `find_positive_label()` finds `Q` or `E_` columns.

3. **Convert times to frames**
   - `time_2_frame()` pads and converts to frame indices.

4. **Adaptive segment sizing**
   - Uses `features.test_seglen_len_lim` and `features.test_hoplen_fenmu` to control segment lengths and hop sizes.

5. **Negative segment construction**
   - Uses labeled gaps as negatives.
   - If negative regions are too short, it estimates negatives from energy (`negative_onset_offset_estimate*`).

6. **Dual-segmentation for post-processing**
   - Builds both adaptive-length segments and short segments for improved negative sampling.

7. **Return test batch**
   - Returns both normal and negative variants plus hop sizes and metadata.

## 8) Minimum Required Files

Before training or evaluation, each `.wav` must have matching `*_logmel.npy` and/or `*_pcen.npy` files.

Typical layout:

```
Training_Set/CLASS_A/audio_001.wav
Training_Set/CLASS_A/audio_001_logmel.npy
Training_Set/CLASS_A/audio_001_pcen.npy
Training_Set/CLASS_A/audio_001.csv
```

## 9) How CSVs Drive the Workflow

CSV files define the event boundaries used to slice feature arrays:

- Rows marked `POS` define positive segments.
- Gaps between positives become negative segments.
- The first `n_shot` positives in eval CSVs can be used as support examples.
- All conversions from seconds to frames add a 25 ms margin to start/end boundaries.

These rules are implemented directly in the dataset classes described above.
