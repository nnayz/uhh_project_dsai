# PREPROCESSING COMPARISON: Original DCASE vs Current Implementation

## Executive Summary

The current implementation significantly **refactors and improves** the original DCASE preprocessing pipeline. While the core audio feature extraction (WAV → log-mel/PCEN) remains similar, there are **major architectural differences** in:

1. **Feature Storage Strategy**: Pre-computed `.npy` files vs. on-the-fly computation
2. **Segmentation Approach**: Random cropping vs. fixed-length padding
3. **Dataset Architecture**: Flat event-based vs. sequence-based dynamic arrays
4. **Configuration Management**: Hydra-based configs vs. YAML/Python dicts
5. **Normalization**: Global statistics vs. per-feature computation

---

## 1. CORE FEATURE EXTRACTION (SIMILAR)

### Original DCASE Approach
**Files:** `baselines/dcase2024_task5/src/datamodules/components/pcen.py`

```python
# Audio Loading & Feature Extraction
y, fs = librosa.load(audio_path, sr=22050)
mel_spec = librosa.feature.melspectrogram(
    y, sr=22050, n_fft=1024, hop_length=256, n_mels=128, fmax=11025
)
logmel = np.log(mel_spec + eps)
pcen = librosa.core.pcen(mel_spec, sr=22050)
features = pcen.T  # Transpose to (time, frequency)
```

### Current Implementation
**Files:** `preprocessing/preprocess.py`

```python
# Same core steps but modularized
def waveform_to_logmel(waveform, cfg):
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    logmel = np.log(mel + eps)
    return logmel.astype(np.float32)

def waveform_to_pcen(waveform, cfg):
    mel = librosa.feature.melspectrogram(...)
    pcen = librosa.pcen(mel, sr=sr, hop_length=hop_length)
    return pcen.astype(np.float32)
```

### Key Difference ✓
- ✓ **Current is cleaner**: Separate functions for each feature type
- ✓ **Current supports both**: Log-mel OR PCEN (configurable)
- ✓ **Current normalizes waveform**: `waveform = waveform / max(waveform)` before processing
- ✓ **Same parameters**: sr=22050, n_fft=1024, hop_mel=256, n_mels=128

---

## 2. FEATURE STORAGE STRATEGY (MAJOR DIFFERENCE!!!!)

### Original DCASE Approach
**Files:** `baselines/dcase2024_task5/src/datamodules/components/feature_extract.py`

```
STRATEGY: Pre-compute and cache to HDF5
├─ Compute PCEN for all audio files (one-time)
├─ Store in HDF5 file structure
│  └─ hf['features'][file_index] = pcen_patch  # Shape: (17, 128)
│  └─ hf['labels'][file_index] = class_label
├─ During training: load from HDF5 (disk → RAM)
└─ Pros: Fast training iteration (already computed)
        Cons: Huge disk space, inflexible
```

**Code Example:**
```python
# Original: Pre-compute and store
for audio_file in all_audio_files:
    y, sr = librosa.load(audio_file, sr=22050)
    pcen = librosa.core.pcen(mel_spec)
    
    # Store segment in HDF5
    hf['features'].resize((index + 1, 17, 128))
    hf['features'][index] = pcen_patch
    hf['labels'][index] = label
    
# During training: just load
feature = hf['features'][idx]  # Already precomputed
```

### Current Implementation
**Files:** `preprocessing/feature_export.py`, `preprocessing/preprocess.py`

```
STRATEGY: Pre-compute to .npy files, load on-demand
├─ Export phase: for each WAV file
│  └─ Compute logmel/PCEN and save as:
│     ├─ audio.wav → audio_logmel.npy  (shape: 128, time_steps)
│     └─ audio.wav → audio_pcen.npy    (shape: 128, time_steps)
├─ During training: load .npy → extract segment → normalize
│  └─ Pros: Flexible, disk-efficient, configurable feature types
│  └─ Cons: Slower loading (disk I/O during training)
└─ Key feature: Segment extraction happens at LOAD TIME
```

**Code Example:**
```python
# Current: Export features once
def export_features(cfg):
    for wav_path in all_wav_files:
        waveform, sr = load_audio(wav_path, cfg)
        logmel = waveform_to_logmel(waveform, cfg)
        pcen = waveform_to_pcen(waveform, cfg)
        
        np.save(wav_path.replace('.wav', '_logmel.npy'), logmel)
        np.save(wav_path.replace('.wav', '_pcen.npy'), pcen)

# During training: load and segment
def extract_logmel_segment(wav_path, start_time, end_time, cfg):
    waveform, sr = load_audio(wav_path, cfg)
    # Load entire feature file
    logmel = np.load(wav_path.replace('.wav', '_logmel.npy'))
    # Extract time segment
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    segment = logmel[:, start_frame:end_frame]
    return segment
```

---

## 3. SEGMENTATION STRATEGY (MAJOR DIFFERENCE 🔴)

### Original DCASE: Fixed-Length with Tiling
**Code from:** `baselines/dcase2024_task5/src/datamodules/components/dynamic_pcen_dataset.py`

```python
def select_segment(self, start, end, pcen, seg_len=17):
    """
    Extract fixed-length segment (17 frames)
    If shorter: TILE/REPEAT the segment
    If longer: random crop
    """
    start_frame = int(start * self.fps)
    end_frame = int(end * self.fps)
    duration_frames = end_frame - start_frame
    
    if duration_frames < seg_len:
        # SHORT: Repeat/tile the segment
        repeat_num = int(seg_len / duration_frames) + 1
        x = np.tile(pcen[:, start_frame:end_frame], (1, repeat_num))
        x = x[:, 0:seg_len]
        return x
    else:
        # LONG: Random crop
        rand_start = np.random.randint(0, duration_frames - seg_len + 1)
        return pcen[:, start_frame + rand_start : start_frame + rand_start + seg_len]
```

**Example:**
```
Annotation: 100-110 frames (10 frames, too short)
seg_len: 17 frames

Original DCASE:
  ├─ Tile twice: [100-110, 100-110, ...]
  ├─ Trim to 17: [100-110, 100-110, 100-106]
  └─ Output shape: (128, 17)

Annotation: 100-200 frames (100 frames, too long)
  ├─ Random start: 50 (between 0 and 83)
  ├─ Extract: [100+50:100+50+17] = [150:167]
  └─ Output shape: (128, 17)
```

### Current Implementation: Padding/Cropping (NEW APPROACH)
**Code from:** `preprocessing/dataset.py`, `preprocessing/preprocess.py`

```python
def crop_pad(t: torch.Tensor, T_max: int) -> torch.Tensor:
    """Crop or pad tensor to fixed time dimension."""
    T = t.shape[-1]
    if T > T_max:
        # Crop: take first T_max frames
        t = t[..., :T_max]
    elif T < T_max:
        # Pad: zero-padding at the end
        diff = T_max - T
        t = F.pad(t, (0, diff))
    return t

# In FewShotEpisodeDataset:
def extract_logmel_segment(wav_path, start_time, end_time, cfg):
    segment = waveform[start_sample:end_sample]
    
    # Pad short segments with zeros
    if min_duration is not None:
        min_samples = int(min_duration * sr)
        if len(segment) < min_samples:
            pad_width = min_samples - len(segment)
            segment = np.pad(segment, (0, pad_width), mode="constant")
```

**Example:**
```
Annotation: 100-110 frames (10 frames, shorter than max_frames)
max_frames: 256 (or T_max)

Current approach:
  ├─ Load segment: shape (128, 10)
  ├─ Pad with zeros: (128, 10) → (128, 256)
  └─ Output: (128, 256) with 246 zero-padded frames

Annotation: 100-200 frames (100 frames)
  ├─ Load segment: shape (128, 100)
  ├─ Crop: take first T_max=256? No, 100 < 256, so pad
  └─ Output: (128, 256)
```

### Key Differences 🔴

| Aspect | Original DCASE | Current |
|--------|----------------|---------|
| **Short segments** | TILE/REPEAT | ZERO-PAD at end |
| **Long segments** | RANDOM CROP | CROP from START |
| **Output shape** | Fixed (17, 128) | Variable (T_max, 128) |
| **Fixed length** | Yes (17 frames) | Flexible T_max (config) |
| **Artificial data?** | Yes (tiling introduces artifacts) | More realistic (padding) |
| **Why?** | Simple episodic training | Variable-length sequences |

---

## 4. DATASET ARCHITECTURE (MAJOR DIFFERENCE 🔴)

### Original DCASE: Sequence-Based Dynamic Arrays
**Files:** `preprocessing/sequence_data/dynamic_pcen_dataset.py`

```
PrototypeDynamicArrayDataSet
├─ Loads pre-computed PCEN features from .npy files
├─ Each __getitem__ returns one SEGMENT (17 frames × 128 bins)
├─ Creates EPISODIC batches via IdentityBatchSampler
│  ├─ Samples k_way=5 classes
│  ├─ Samples n_shot=5 examples per class (support)
│  ├─ Samples n_query=5 examples per class (query)
│  └─ Total batch: 50 examples
└─ Trains with prototypical loss (metric learning)
```

**Code:**
```python
class PrototypeDynamicArrayDataSet(Dataset):
    def __getitem__(self, idx):
        class_name = self.classes[idx]
        segment = self.select_positive(class_name)  # (17, 128)
        return segment.astype(np.float32), self.classes2int[class_name]
```

### Current Implementation: Flat Event-Based OR Dynamic Arrays
**Files:** `preprocessing/dataset.py` (New), `preprocessing/datamodule.py`

**Two dataset paths:**

#### Path A: Flat DCASEEventDataset (NEW)
```
DCASEEventDataset
├─ Loads all labeled segments (events) from CSV annotations
├─ Each __getitem__ returns:
│  └─ Tensor shape (1, n_mels, T)  where T varies per example
├─ Entire annotation as one example (NOT segmented into fixed 17 frames)
├─ Wrapped in FewShotEpisodeDataset for episodic training
│  ├─ Creates episodes on-the-fly
│  ├─ Crops/pads to fixed T_max
│  └─ Returns support/query sets
└─ More flexible: works with variable-length annotations
```

**Code:**
```python
class DCASEEventDataset(Dataset):
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        ex = self.examples[idx]
        logmel = extract_logmel_segment(
            wav_path=ex.wav_path,
            start_time=ex.start_time,
            end_time=ex.end_time,
            cfg=self.cfg,
        )
        tensor = torch.from_numpy(logmel)[None, ...]  # (1, n_mels, T)
        label = ex.class_id
        return tensor, label

class FewShotEpisodeDataset(Dataset):
    def __getitem__(self, idx):
        # Sample k_way classes, n_shot support, n_query query
        # Crop/pad all to T_max
        # Return (support_x, support_y, query_x, query_y)
```

#### Path B: PrototypeDynamicArrayDataSet (ORIGINAL, still used)
```
Same as original DCASE
├─ Dynamic pre-computed PCEN features
├─ Fixed-length segments (17 frames)
├─ Episodic batching via IdentityBatchSampler
└─ Used when config specifies this dataset
```

---

## 5. CONFIGURATION MANAGEMENT (MODERATE DIFFERENCE)

### Original DCASE
**Files:** Various YAML files in `conf/` + Python dicts

```yaml
# Config scattered across multiple files
features:
  seg_len: 0.200          # Seconds
  hop_seg: 0.100          # Seconds
  sr: 22050
  n_fft: 1024
  hop_mel: 256
  n_mels: 128
  fmax: 11025

train_param:
  n_shot: 5
  k_way: 5
  negative_train_contrast: false
```

### Current Implementation
**Files:** `conf/config.yaml` (unified Hydra config)

```yaml
# Single unified config with Hydra
features:
  eps: 1e-8
  fmax: 11025
  fmin: 50
  sr: 22050
  n_fft: 1024
  n_mels: 128
  hop_mel: 256
  feature_types: logmel  # Can be "logmel" or "pcen" or "logmel@pcen"
  embedding_dim: 2048

train_param:
  seg_len: 0.2
  n_shot: 5
  k_way: 5
  adaptive_seg_len: false  # NEW: variable-length testing

annotations:
  min_duration: 0.2      # NEW: minimum segment duration
  max_frames: 256        # NEW: maximum frames for padding
  positive_label: "POS"
  class_name: "Class"
```

**Key additions:**
- ✓ `feature_types`: Configurable (logmel, pcen, or both)
- ✓ `adaptive_seg_len`: Variable-length testing
- ✓ `max_frames`: Fixed padding dimension
- ✓ `positive_label`: Flexible annotation parsing

---

## 6. NORMALIZATION & STATISTICS (MODERATE DIFFERENCE)

### Original DCASE
**Files:** `baselines/dcase2024_task5/src/datamodules/components/pcen.py`

```python
class Feature_Extractor:
    mean_std = {}  # Class variable
    
    def update_mean_std(self):
        """Compute global mean/std for each feature type"""
        for suffix in self.feature_types:
            features = []
            for audio_path in tqdm(self.files[:1000]):  # ~1000 files
                feature_path = audio_path.replace(".wav", f"_{suffix}.npy")
                features.append(np.load(feature_path).flatten())
            
            all_data = np.concatenate(features)
            mean = np.mean(all_data)  # Single value
            std = np.std(all_data)    # Single value
            Feature_Extractor.mean_std[suffix] = [mean, std]
    
    def extract_feature(self, audio_path, normalized=True):
        feat = np.load(...)
        if normalized:
            mean, std = Feature_Extractor.mean_std[suffix]
            feat = (feat - mean) / std  # Z-score
        return feat
```

**Stored statistics (for DCASE AudioMNIST):**
```
mean = 1.4421
std = 1.2201
```

### Current Implementation
**Status: NOT explicitly shown in current code**

- The current code loads features but **normalization approach is not clear**
- Likely uses the same global statistics from original
- OR computes on-the-fly during training (not shown in provided files)

---

## 7. DATA AUGMENTATION (NEW FEATURES)

### Original DCASE
```python
# Minimal augmentation
# - Tiling (handled as segment extraction)
# - Optional mixing with negative samples (commented out)
```

### Current Implementation
**New capabilities in `dynamic_pcen_dataset.py`:**

```python
# Optional negative contrast learning
if self.train_param.negative_train_contrast:
    segment_neg = self.select_negative(class_name)
    return (
        segment.astype(np.float32),
        segment_neg.astype(np.float32),
        self.classes2int[class_name] * 2,
        self.classes2int[class_name] * 2 + 1,
    )

# Optional adaptive segment length for testing
if self.train_param.adaptive_seg_len:
    self.data_test = PrototypeAdaSeglenBetterNegTestSetV2(...)
else:
    self.data_test = PrototypeTestSet(...)
```

**New augmentations:**
- ✓ Negative contrast pairs
- ✓ Adaptive segment length for evaluation
- ✓ Better negative sampling strategy

---

## 8. ANNOTATION PARSING (NEW)

### Original DCASE
**Implicit CSV format handling**
```python
# Assumed specific CSV structure:
# Audiofilename, Starttime, Endtime, ...
```

### Current Implementation
**Explicit annotation service:**

**Files:** `preprocessing/ann_service.py`, `schemas/segment_example.py`

```python
class AnnotationService:
    """
    Handles multiple CSV formats:
    1. Multi-class CSVs with CLASS_x columns (POS/NEG/UNK)
    2. Single-class CSVs with 'Q' column (POS/UNK)
    3. Fallback: only Audiofilename/Starttime/Endtime (all positive)
    """
    
    def load_annotations(self, annotation_paths):
        # Parse different CSV formats
        # Create SegmentExample objects
        return examples  # List of SegmentExample

class SegmentExample:
    """Standardized annotation format"""
    wav_path: Path
    start_time: float
    end_time: float
    class_name: str
    class_id: int
```

**Benefits:**
- ✓ Handles multiple annotation formats
- ✓ Standardized data structures
- ✓ Better error reporting

---

## 9. SUMMARY TABLE

| Aspect | Original DCASE | Current | Impact |
|--------|---|---|---|
| **Feature Extraction** | Log-mel + PCEN | Log-mel + PCEN (configurable) | ✓ More flexible |
| **Feature Storage** | HDF5 (pre-computed) | .npy files (per-audio) | ✓ More disk-efficient, flexible |
| **Segmentation** | Fixed 17 frames + tiling | Variable T_max + padding | 🔴 Different training dynamics |
| **Short segment handling** | TILE/REPEAT | ZERO-PAD | 🔴 Less artificial data |
| **Long segment handling** | RANDOM CROP | CROP from START | 🟡 Different sampling |
| **Dataset class** | PrototypeDynamicArrayDataSet | DCASEEventDataset OR PrototypeDynamicArrayDataSet | ✓ More options |
| **Episodic batching** | IdentityBatchSampler | Same IdentityBatchSampler | ✓ Compatible |
| **Config system** | Scattered YAML + dicts | Hydra (unified) | ✓ Better management |
| **Annotation parsing** | Implicit CSV | Explicit AnnotationService | ✓ More robust |
| **Augmentation** | Minimal | Negative contrast, adaptive seg-len | ✓ Enhanced |

---

## 10. WHICH DATASET ARE WE USING?

### For Training:
**Default:** `PrototypeDynamicArrayDataSet` (original DCASE)
```python
if self.train_param.use_validation_first_5:
    self.dataset = PrototypeDynamicArrayDataSetWithEval(...)
else:
    self.dataset = PrototypeDynamicArrayDataSet(...)  # Default
```

**Optional:** `DCASEEventDataset + FewShotEpisodeDataset` (new flat approach)
```python
# Create flat dataset
base_dataset = DCASEEventDataset(annotations=[...], cfg=cfg)
# Wrap for episodic training
episode_dataset = FewShotEpisodeDataset(base_dataset, cfg)
```

### For Validation:
**Primary:** `PrototypeDynamicArrayDataSetVal`
```python
self.val_dataset = PrototypeDynamicArrayDataSetVal(...)
```

### For Testing:
**Default:** `PrototypeTestSet`
```python
if self.train_param.adaptive_seg_len:
    self.data_test = PrototypeAdaSeglenBetterNegTestSetV2(...)
else:
    self.data_test = PrototypeTestSet(...)
```

---

## 11. KEY ARCHITECTURAL CHANGES

### 1. **Modularization** ✓
- Original: Monolithic feature extraction in one class
- Current: Separated into `load_audio()`, `waveform_to_logmel()`, `waveform_to_pcen()`, `extract_logmel_segment()`

### 2. **Flexibility** ✓
- Original: Fixed to PCEN, HDF5 storage, 17-frame segments
- Current: Configurable features, .npy storage, variable-length support

### 3. **Dual-path support** ✓
- Original: Single PrototypeDynamicArrayDataSet
- Current: Can use either original sequence-based OR new flat event-based

### 4. **Better config management** ✓
- Original: Multiple config files scattered
- Current: Unified Hydra configuration

### 5. **Production-ready annotations** ✓
- Original: Assumes specific CSV format
- Current: Robust AnnotationService handling multiple formats

---

## 12. POTENTIAL ISSUES & CONSIDERATIONS

### Issue 1: Padding vs. Tiling
**Original (tiling):** Artificially extends short events by repeating them
```
Short event: [A, B, C] → [A, B, C, A, B, C, A, B, C, ...]
```

**Current (padding):** Extends with zeros
```
Short event: [A, B, C] → [A, B, C, 0, 0, 0, 0, ...]
```

**Impact:** Model trained on tiled data may not generalize well to padded data and vice versa. **This is a breaking change** if you're reusing pre-trained weights.

### Issue 2: Segment Extraction Timing
**Original:** Segments extracted during preprocessing → fixed 17 frames
**Current:** Can extract at load time → variable lengths possible

**Impact:** Current is more flexible but requires managing variable-length sequences in the model.

### Issue 3: Feature Storage Size
**Original HDF5:**
```
1000 audio files × 430 timesteps × 128 bins × 4 bytes (float32)
= ~220 MB HDF5 file (with compression)
```

**Current .npy files:**
```
1000 separate files × 128 × T frames × 4 bytes
= ~220 MB total (more fragmented)
```

**Impact:** Current is slightly less efficient for disk I/O but more flexible.

---

## 13. RECOMMENDED APPROACH

### Use `PrototypeDynamicArrayDataSet` (Current Default) if:
- ✓ You want to match original DCASE behavior exactly
- ✓ Pre-computed features work for your setup
- ✓ Fixed-length 17-frame segments are sufficient
- ✓ You're continuing from a pre-trained model

### Use `DCASEEventDataset + FewShotEpisodeDataset` if:
- ✓ You have variable-length annotations
- ✓ You want more flexible segment handling
- ✓ You're starting fresh (no pre-trained weights)
- ✓ You prefer padding over tiling
- ✓ You want better code modularity

---

## 14. IMPLEMENTATION PRIORITY

To fully understand what's different:

1. **Check config.yaml** → What feature_types are we using? (logmel vs pcen)
2. **Check datamodule.py** → Which dataset class is instantiated?
3. **Check main.py** → How is preprocessing called?
4. **Check archs/*/lightning_module.py** → How does the model handle variable-length inputs?

