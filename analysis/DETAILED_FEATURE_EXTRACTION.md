# DETAILED FEATURE EXTRACTION WALKTHROUGH

## Overview
The feature extraction process converts raw WAV audio files into PCEN (Per-Channel Energy Normalization) features that can be fed to the neural network. This document provides step-by-step computational details.

---

## PHASE 1: RAW AUDIO LOADING

### Step 1.1: Load WAV File
**Code Location:** `baselines/deep_learning/Feature_extract.py` (line 116-118)
```python
def extract_feature(audio_path, feat_extractor, conf):
    y, fs = librosa.load(audio_path, sr=conf.features.sr)
    # y,fs = librosa.load(audio_path,sr=22050)
```

**What happens:**
- `librosa.load()` reads the WAV file from disk
- Audio is resampled to `sr=22050 Hz` (standard sample rate for speech/audio)
- Output: `y` = array of float values (audio samples), `fs` = actual sample rate

**Example:**
```
Input:  audio.wav (44100 Hz stereo, 5 seconds)
        ├─ Read and convert to mono
        └─ Resample from 44100 Hz → 22050 Hz
Output: y = array of shape (110250,) = 22050 * 5
        fs = 22050
```

### Step 1.2: Audio Scaling
**Code Location:** `baselines/deep_learning/Feature_extract.py` (line 119-120)
```python
y = y * (2**32)
# Scale audio by 2^32 = 4,294,967,296
pcen = feat_extractor.extract_feature(y)
```

**Why?**
- Librosa.pcen() expects audio in a specific range for optimal numerical computation
- Scaling by 2^32 converts normalized [-1, 1] range to larger values
- Helps with numerical stability in PCEN computation

**Example:**
```
Before:  y = [-0.5, -0.25, 0.0, 0.25, 0.5]
After:   y = [-2147483648, -1073741824, 0, 1073741824, 2147483648]
```

---

## PHASE 2: MEL-SPECTROGRAM EXTRACTION

### Step 2.1: Compute Mel-Spectrogram
**Code Location:** `baselines/deep_learning/Feature_extract.py` (line 106-111)
```python
class Feature_Extractor():
    def extract_feature(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            audio,
            sr=self.sr,              # 22050 Hz
            n_fft=self.n_fft,        # 1024
            hop_length=self.hop,     # 256
            n_mels=self.n_mels,      # 128
            fmax=self.fmax           # 11025
        )
        # mel_spec is now shape (128, time_steps)
```

**Detailed breakdown of what `melspectrogram()` does:**

#### 2.1.1: Windowing
```
Step 1: Divide audio into overlapping frames
┌─────────────────────────────────────────────────────────┐
│ Audio signal: 110250 samples (5 seconds @ 22050 Hz)    │
└─────────────────────────────────────────────────────────┘

Frame parameters:
  - n_fft = 1024 samples
  - hop_length = 256 samples
  - Window function = Hann window (default)
  
Number of frames:
  n_frames = (110250 - 1024) / 256 + 1 = 430 frames
  Time resolution: 256 / 22050 = 11.6 ms per frame
```

**Visual:**
```
Audio timeline (0 to 5 seconds):
│●●●●●│●●●●●│●●●●●│...
└─Frame 1──┘
    └─Frame 2──┘
        └─Frame 3──┘

Frame 1: samples [0:1024]
Frame 2: samples [256:1280]
Frame 3: samples [512:1536]
...
```

#### 2.1.2: FFT Computation
```
Step 2: Apply FFT to each frame
For each frame:
  1. Apply Hann window function
  2. Compute FFT (1024-point Fast Fourier Transform)
  3. Take magnitude: |FFT| = sqrt(real² + imag²)
  4. Compute power: power_spec = |FFT|² / n_fft
  
Output: Power spectrogram
  Shape: (1024 frequency bins, 430 time steps)
  Range: [0, max_power] (non-negative)
```

#### 2.1.3: Mel-Scale Filterbank
```
Step 3: Convert linear frequency scale to mel-scale
Linear frequency (Hz):
  ├─ 0 Hz      (no sound)
  ├─ 100 Hz    (low frequency)
  ├─ 5000 Hz   (mid frequency)
  ├─ 11025 Hz  (high frequency - fmax)
  └─ 11050 Hz  (Nyquist frequency = sr/2)

Mel-scale (perceptual):
  ├─ 0 Mel
  ├─ 200 Mel    (≈ 100 Hz)
  ├─ 2000 Mel   (≈ 5000 Hz)
  ├─ 3000 Mel   (≈ 11025 Hz)
  └─ (non-linear mapping)

Why Mel-scale?
  - Human ear perception is non-linear
  - We are better at distinguishing between 100-200 Hz
    than 10000-10100 Hz
  - Mel-scale gives more resolution at lower frequencies
```

**128 Mel-filters:**
```
128 triangular filters distributed across mel-scale
Filter 1:   covers ~0-150 Hz
Filter 64:  covers ~5000-6000 Hz
Filter 128: covers ~10000-11025 Hz

Each filter multiplies power spectrum and sums the result:
  mel_spec[i, t] = sum(power_spec[:, t] * filter[i])
```

**Example output after mel-filterbank:**
```
mel_spec shape: (128, 430)
  - 128 frequency bins (mel-scale)
  - 430 time steps
  
mel_spec values: mostly in range [0.1, 10000] (power values)
```

---

## PHASE 3: PCEN TRANSFORMATION

### Step 3.1: Apply PCEN
**Code Location:** `baselines/deep_learning/Feature_extract.py` (line 112-114)
```python
pcen = librosa.core.pcen(mel_spec, sr=22050)
pcen = pcen.astype(np.float32)
return pcen
```

**What is PCEN?**
PCEN = Per-Channel Energy Normalization

It applies temporal smoothing and normalization to each mel-channel independently.

**Mathematical formula (simplified):**
```
pcen[i, t] = (S[i, t] / (M[i, t] + epsilon))^alpha - delta

Where:
  S[i, t]     = mel_spec[i, t] (input power)
  M[i, t]     = moving average of S[i, t] (temporal context)
  epsilon     = small constant to avoid division by zero
  alpha       = 0.98 (default compression factor)
  delta       = 2.0 (default bias)

M[i, t] = (1 - decay) * M[i, t-1] + decay * S[i, t]
  decay = 0.01 (default temporal decay rate)
```

**Intuition:**
```
For each frequency bin (mel-channel):
  1. Track the running average energy (moving average)
  2. Normalize current energy by this average
  3. Apply compression and bias
  4. Result: energy normalized relative to local context

Example for one frequency bin over time:
  
Time:      1    2    3    4    5    6    7
Power:   100  110  120  500  510  520  130
Moving:  100  105  112  240  300  350  260
PCEN:    -1.5 -1.2 -0.8  1.2  1.0  0.9 -0.8

Notice: PCEN responds to relative changes, not absolute values
        Constant 500-520 power gives low PCEN
        Because it's expected based on recent history
```

**Output:**
```
pcen shape: (128, 430) - same as mel_spec
pcen range: typically [-2 to 2] (normalized)
dtype: float32
```

**Why PCEN for Bioacoustics?**
```
Scenario: Bird call at 5000 Hz with wind noise background

Traditional log-mel-spectrogram:
  ├─ Captures: loud bird call, loud wind noise
  ├─ Problem: both appear equally loud
  └─ Model gets confused

PCEN:
  ├─ Tracks: baseline wind noise energy
  ├─ When bird call appears: sudden increase above baseline
  ├─ Result: bird call stands out as deviation
  └─ Model learns: "increases above local baseline = event"
  
PCEN is adaptive to changing background noise!
```

---

## PHASE 4: TRANSPOSE & TYPE CONVERSION

### Step 4.1: Transpose
**Code Location:** `baselines/deep_learning/Feature_extract.py` (line 121)
```python
return pcen.T  # Transpose!
```

**Why transpose?**
```
Before:  pcen.shape = (128, 430)
         └─ 128 frequency bins (rows)
         └─ 430 time steps (columns)
         └─ This is "frequency-first" format

After:   pcen.T.shape = (430, 128)
         └─ 430 time steps (rows)
         └─ 128 frequency bins (columns)
         └─ This is "time-first" format

Why time-first?
  - PyTorch conventions expect (sequence_length, features)
  - CNN input needs (batch, channels, height, width)
  - But here, sequential models expect time as first dimension
```

**Output:**
```
Shape: (430, 128)
  - 430 time steps
  - 128 mel-frequency features per timestep
```

### Step 4.2: Data Type Conversion
```python
pcen = pcen.astype(np.float32)
```

**Why float32?**
```
float64 (default):  8 bytes per value, 430 * 128 = 55,040 values
                    = 440 KB per audio (memory heavy)
float32:            4 bytes per value, same 440 KB becomes 220 KB
                    Still precise enough for audio features
                    GPU compatibility (most GPUs optimize for float32)
```

---

## PHASE 5: SEGMENTATION & STORAGE (Training Pipeline)

### Step 5.1: Convert Time to Frames
**Code Location:** `baselines/deep_learning/Feature_extract.py` (line 139-153)
```python
def time_2_frame(df, fps):
    '''Convert annotation times to frame indices'''
    # Add 25ms margin around onset/offset
    df.loc[:, 'Starttime'] = df['Starttime'] - 0.025
    df.loc[:, 'Endtime'] = df['Endtime'] + 0.025
    
    # Convert seconds to frame indices
    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]
    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]
    
    return start_time, end_time

# Example:
# fps = sr / hop_length = 22050 / 256 = 86.13 frames/second
# Annotation: "Bird call from 1.0 to 2.5 seconds"
# After margin: 0.975 to 2.525 seconds
# Convert to frames:
#   start = floor(0.975 * 86.13) = 84 frames
#   end = floor(2.525 * 86.13) = 217 frames
```

### Step 5.2: Segment Extraction (Long Annotations)
**Code Location:** `baselines/deep_learning/Feature_extract.py` (line 57-69)
```python
if end_ind - str_ind > seg_len:
    # Annotation is longer than segment length
    # Extract multiple overlapping segments
    
    shift = 0
    while end_ind - (str_ind + shift) > seg_len:
        # Extract segment from [str_ind+shift : str_ind+shift+seg_len]
        pcen_patch = pcen[int(str_ind + shift):int(str_ind + shift + seg_len)]
        
        # Store in HDF5 file
        hf['features'].resize((file_index + 1, pcen_patch.shape[0], pcen_patch.shape[1]))
        hf['features'][file_index] = pcen_patch
        label_list.append(label)
        file_index += 1
        shift = shift + hop_seg  # Move by hop_seg frames
```

**Visual example:**
```
Annotation: bird call from frame 84 to 400
seg_len = 17 frames, hop_seg = 10 frames

Extract segments:
  Segment 1: frames [84 : 101]      (17 frames)
  Segment 2: frames [94 : 111]      (17 frames, shifted by 10)
  Segment 3: frames [104 : 121]     (17 frames, shifted by 10)
  ...
  Segment N: frames [383 : 400]     (last segment)

Result:
  - Multiple training examples from single annotation
  - Each sample: (17, 128) = 17 time steps × 128 mel-bins
  - Same label for all segments
```

### Step 5.3: Segment Extraction (Short Annotations)
**Code Location:** `baselines/deep_learning/Feature_extract.py` (line 78-92)
```python
else:
    # Annotation is shorter than segment length
    # Tile/repeat to reach required length
    
    pcen_patch = pcen[str_ind:end_ind]
    repeat_num = int(seg_len / pcen_patch.shape[0]) + 1
    pcen_patch_new = np.tile(pcen_patch, (repeat_num, 1))
    pcen_patch_new = pcen_patch_new[0:int(seg_len)]
```

**Visual example:**
```
Annotation: bird call from frame 100 to 110 (only 10 frames)
seg_len = 17 frames (required)

Step 1: Extract segment
  pcen_patch = pcen[100:110]  # Shape: (10, 128)

Step 2: Calculate repeat count
  repeat_num = int(17 / 10) + 1 = 2
  
Step 3: Tile the segment
  [original] [original]
  Frames:  100-110, 100-110
  
Step 4: Trim to required length
  Take first 17 frames: frames 100-110, then 100-106
  Result shape: (17, 128)

Result:
  Segment = [bird_call_frames, bird_call_frames, partial_bird_call]
  └─ Artificially extended to match fixed length
  └─ Model learns from repeated pattern
```

---

## PHASE 6: NORMALIZATION (Z-score Standardization)

### Step 6.1: Compute Global Mean & Std
**Code Location:** `baselines/dcase2024_task5/src/datamodules/components/pcen.py` (line 42-54)
```python
def update_mean_std(self, feature_types=None):
    """Calculate global statistics across dataset"""
    if len(Feature_Extractor.mean_std.keys()) != 0:
        return  # Already computed, skip
    
    print("Calculating mean and std")
    for suffix in self.feature_types:  # e.g., "pcen"
        print(f"Calculating: {suffix}")
        features = []
        
        # Load pre-computed features from ~1000 audio files
        for audio_path in tqdm(self.files[:1000]):
            feature_path = audio_path.replace(".wav", f"_{suffix}.npy")
            features.append(np.load(feature_path).flatten())  # Flatten to 1D
        
        # Concatenate all features
        all_data = np.concatenate(features)  # Shape: (num_samples,)
        
        # Compute statistics
        Feature_Extractor.mean_std[suffix] = [
            np.mean(all_data),      # Global mean
            np.std(all_data)        # Global standard deviation
        ]
```

**Numerical example:**
```
Load 1000 audio files, each with PCEN shape (time_steps, 128)

File 1: (430, 128) → flatten → 55,040 values
File 2: (425, 128) → flatten → 54,400 values
...
File 1000: (440, 128) → flatten → 56,320 values

Concatenate: all_data with shape (~50 million values,)

Compute:
  mean = sum(all_data) / len(all_data) ≈ 1.4421
  std = sqrt(sum((all_data - mean)²) / len(all_data)) ≈ 1.2201
  
Store globally:
  Feature_Extractor.mean_std['pcen'] = [1.4421, 1.2201]
```

### Step 6.2: Apply Z-score Normalization
**Code Location:** `baselines/dcase2024_task5/src/datamodules/components/pcen.py` (line 57-68)
```python
def extract_feature(self, audio_path, feature_types=None, normalized=True):
    """Load and normalize features"""
    features = []
    for suffix in self.feature_types:
        feature_path = audio_path.replace(".wav", f"_{suffix}.npy")
        feat = np.load(feature_path)  # Load pre-computed features
        
        if normalized:
            mean, std = Feature_Extractor.mean_std[suffix]
            # Z-score normalization: (X - μ) / σ
            feat = (feat - mean) / std
        
        features.append(feat)
    
    # Concatenate features along frequency dimension
    return np.concatenate(features, axis=1)
```

**Formula:**
```
X_normalized = (X - μ) / σ

Where:
  X = original feature value
  μ = global mean (1.4421)
  σ = global standard deviation (1.2201)
  X_normalized = z-score (standardized value)
```

**Numerical example:**
```
Original PCEN value: 2.5
Global mean:        1.4421
Global std:         1.2201

X_normalized = (2.5 - 1.4421) / 1.2201
             = 1.0579 / 1.2201
             = 0.867

Original value:     -0.5
X_normalized = (-0.5 - 1.4421) / 1.2201
             = -1.9421 / 1.2201
             = -1.592

Result:
  ├─ Values centered around 0
  ├─ Std deviation ≈ 1
  ├─ Range typically [-3, 3]
  └─ More stable for neural network training
```

**Why normalization?**
```
Benefits:
  1. Neural networks train faster with zero-mean inputs
  2. Gradient flow is more stable
  3. Learning rate doesn't need adjustment for different ranges
  4. Prevents one feature from dominating others

Before normalization:
  Features in range [0, 5] → network learns slowly
  
After normalization:
  Features in range [-3, 3] → network converges quickly
```

---

## PHASE 7: BATCH CREATION FOR TRAINING

### Step 7.1: Episodic Batching
**Code Location:** `baselines/dcase2024_task5/src/datamodules/components/batch_sampler.py`

**Prototype Networks training strategy:**
```
Create episodic batches:
  - k_way = 5 (number of classes)
  - n_shot = 5 (support examples per class)
  - Query examples = same number
  
Batch composition:
  Support set:  5 classes × 5 examples = 25 samples
  Query set:    5 classes × 5 examples = 25 samples
  Total batch:  50 samples
  
Each sample shape: (17, 128)
  ├─ 17 time steps
  └─ 128 mel-frequency bins
  
Batch shape: (50, 17, 128)
```

**How batches are created:**
```python
# Episodic sampling:
for episode in range(n_episodes):
    batch = []
    classes = random_sample(all_classes, k=5)  # Select 5 random classes
    
    for cls in classes:
        # Get 10 examples from this class (5 support + 5 query)
        examples = random_sample(cls_examples, k=10)
        batch.extend(examples)
    
    # batch now has 50 examples (5 classes × 10 per class)
    yield batch
```

---

## COMPLETE FEATURE EXTRACTION FLOW (SUMMARY)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LOAD AUDIO
│    audio.wav (44100 Hz) → librosa.load() → (22050 Hz)
│    Output: y = (110250,) audio samples
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 2. SCALE AUDIO
│    y = y * 2^32
│    Output: y = (110250,) with range [-2.1e9, 2.1e9]
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 3. MEL-SPECTROGRAM
│    - Divide into frames (hop=256, n_fft=1024)
│    - FFT on each frame
│    - Apply 128 mel-scale filters
│    Output: mel_spec = (128, 430)
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 4. PCEN TRANSFORMATION
│    - Temporal smoothing per channel
│    - Energy normalization
│    - Compression and bias
│    Output: pcen = (128, 430)
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 5. TRANSPOSE & CONVERT
│    - Transpose to (430, 128)
│    - Convert to float32
│    Output: pcen.T = (430, 128)
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 6. PARSE ANNOTATIONS
│    - Read CSV with onset/offset times
│    - Convert to frame indices (multiply by fps=86.13)
│    - Add 25ms margin
│    Output: start_frames, end_frames
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 7. SEGMENT EXTRACTION
│    If annotation longer than 17 frames:
│      - Extract overlapping 17-frame segments (hop=10)
│    If annotation shorter than 17 frames:
│      - Tile/repeat to 17 frames
│    Output: Multiple segments of shape (17, 128)
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 8. COMPUTE GLOBAL STATISTICS
│    - Load 1000 audio files
│    - Flatten all features
│    - Compute global mean & std
│    Output: mean=1.4421, std=1.2201
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 9. NORMALIZATION (Z-SCORE)
│    feat = (feat - 1.4421) / 1.2201
│    Output: feat ∈ [-3, 3] typically
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 10. BATCH CREATION
│     - Select 5 random classes
│     - Sample 10 examples from each class
│     - Create batch of 50 samples
│     Output: batch.shape = (50, 17, 128)
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│ 11. NEURAL NETWORK INPUT
│     Reshape to (50, 1, 17, 128) for CNN
│     Feed through ResNet encoder
│     Output: embeddings (50, 512)
└─────────────────────────────────────────────────────────────┘
```

---

## KEY PARAMETERS REFERENCE

```yaml
Audio Processing:
  Sample Rate (sr):        22050 Hz
  FFT Window Size (n_fft): 1024 samples
  Hop Length:              256 samples
  
Time Resolution:
  Frame Duration:          1024 / 22050 ≈ 46.4 ms
  Hop Duration:            256 / 22050 ≈ 11.6 ms
  Frames Per Second:       22050 / 256 ≈ 86.13 fps
  
Mel-Spectrogram:
  Number of Mel-Bins:      128
  Max Frequency (fmax):    11025 Hz (= sr/2)
  Min Frequency (fmin):    50 Hz
  
Segmentation:
  Segment Length:          0.2 seconds
  Segment Length (frames): 17 frames
  Hop Segment:             0.1 seconds
  Hop Segment (frames):    10 frames
  
Normalization:
  Global Mean:             1.4421
  Global Std:              1.2201
  
Training:
  Classes Per Episode:     5 (k_way)
  Support Per Class:       5 (n_shot)
  Query Per Class:         5 (n_query)
  Batch Size:              50 samples
```

---

## DEBUGGING: WHAT TO CHECK

If features look wrong, check these steps:

```python
# Step 1: Audio loading
y, sr = librosa.load(path, sr=22050)
assert -1 <= y.min() <= y.max() <= 1  # Should be normalized
assert sr == 22050

# Step 2: Mel-spectrogram
mel = librosa.feature.melspectrogram(y, sr=22050, n_fft=1024, hop_length=256, n_mels=128)
assert mel.shape == (128, ~430)  # 128 mel-bins
assert mel.min() >= 0  # Should be non-negative (power)

# Step 3: PCEN
pcen = librosa.core.pcen(mel)
assert -2 <= pcen.min() <= pcen.max() <= 2  # Should be normalized
assert pcen.shape == mel.shape

# Step 4: Transpose
pcen_t = pcen.T
assert pcen_t.shape == (~430, 128)  # time × frequency

# Step 5: Segmentation
assert segment.shape == (17, 128)  # Fixed size

# Step 6: Normalization
assert -3 <= segment.min() <= segment.max() <= 3  # Z-score range
```
