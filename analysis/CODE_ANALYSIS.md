# DCASE Few-Shot Bioacoustic Code Analysis

## Project Overview
This project implements **few-shot learning for bioacoustic event detection** using prototype networks. The main goal is to detect and classify different animal/bird calls from audio files with minimal training examples.

---

## 1. MOST IMPORTANT CLASSES & KEY FILES

### **Data Processing Pipeline**
| Component | File | Purpose |
|-----------|------|---------|
| **Feature_Extractor** | [src/datamodules/components/pcen.py](baselines/dcase2024_task5/src/datamodules/components/pcen.py) | Extracts PCEN features from raw audio; calculates mean/std for normalization |
| **PrototypeDynamicArrayDataSet** | [src/datamodules/components/dynamic_pcen_dataset.py](baselines/dcase2024_task5/src/datamodules/components/dynamic_pcen_dataset.py) | Main training dataset that loads pre-computed PCEN features dynamically |
| **Datagen** | [src/datamodules/components/Datagenerator.py](baselines/dcase2024_task5/src/datamodules/components/Datagenerator.py) | Handles data loading, class balancing, train/val splitting, feature normalization |

### **Model Training**
| Component | File | Purpose |
|-----------|------|---------|
| **PrototypeModule** | [src/models/prototype_module.py](baselines/dcase2024_task5/src/models/prototype_module.py) | PyTorch Lightning module for training/validation/testing |
| **ResNet / SimpleDenseNet** | [src/models/components/simple_dense_net.py](baselines/dcase2024_task5/src/models/components/simple_dense_net.py) | Encoder network (feature extraction backbone) |
| **BasicBlock** | [baselines/deep_learning/Model.py](baselines/deep_learning/Model.py) | Residual blocks used in ResNet encoder |

### **Loss Functions & Metrics**
| Component | File | Purpose |
|-----------|------|---------|
| **prototypical_loss** | [src/utils/loss.py](baselines/dcase2024_task5/src/utils/loss.py) | Computes prototype loss for few-shot learning |
| **evaluation.py** | [src/utils/evaluation.py](baselines/dcase2024_task5/src/utils/evaluation.py) | Evaluation metrics for onset/offset prediction |

### **Batch Sampling**
| Component | File | Purpose |
|-----------|------|---------|
| **IdentityBatchSampler / EpisodicBatchSampler** | [src/datamodules/components/batch_sampler.py](baselines/dcase2024_task5/src/datamodules/components/batch_sampler.py) | Creates episodic batches for meta-learning |

---

## 2. AUDIO PREPROCESSING PIPELINE

### **Audio Input → PCEN Features Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│                     WAV FILE LOADING                             │
│  librosa.load(audio_path, sr=22050)                             │
│  Output: time-series audio signal (sample_rate=22050 Hz)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              MEL-SPECTROGRAM COMPUTATION                         │
│  librosa.feature.melspectrogram(                                │
│    audio, sr=22050, n_fft=1024, hop_length=256,               │
│    n_mels=128, fmax=11025                                       │
│  )                                                               │
│  Output: (n_mels=128, time_steps) shaped array                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              LOG MEL-SPECTROGRAM                                 │
│  log(mel_spec + eps) where eps=2.22e-16                         │
│  Converts energy to log scale (human perception)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│           PCEN (PER-CHANNEL ENERGY NORMALIZATION)               │
│  librosa.core.pcen(mel_spec, sr=22050)                         │
│  Applies temporal compression and normalization                 │
│  Output: (n_mels=128, time_steps) normalized features           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              TRANSPOSE & TYPE CONVERSION                         │
│  pcen.T  →  (time_steps, 128)                                   │
│  .astype(np.float32)                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│         SEGMENTATION & FEATURE NORMALIZATION                    │
│  - Split into fixed-length segments                             │
│  - Normalize: (X - mean) / std                                   │
│  - mean=1.4421, std=1.2201 (pre-computed)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. DETAILED EXPLANATION: MEL-SPECTROGRAM & LOG-MEL-SPECTROGRAM

### **Step 1: Mel-Spectrogram Extraction**

**Function:** `librosa.feature.melspectrogram()`

**Parameters used:**
```python
sr = 22050         # Sample rate (Hz) - standard for speech/audio
n_fft = 1024       # FFT window size (frequency resolution)
hop_length = 256   # Hop/stride between frames (time resolution)
n_mels = 128       # Number of mel-frequency bins
fmax = 11025       # Maximum frequency (Hz) - Nyquist = 22050/2
```

**What it does:**
1. **Divides audio into overlapping frames** of 1024 samples with 256-sample hop
   - Frame duration: 1024/22050 ≈ 46.4 ms
   - Hop duration: 256/22050 ≈ 11.6 ms
   
2. **Computes FFT** for each frame → magnitude spectrum

3. **Applies Mel-scale filterbank** (128 filters) to map linear frequency to mel-scale
   - **Why Mel-scale?** Human ear perceives frequency logarithmically
   - Lower frequencies get more resolution, higher frequencies get less
   - Example: 100 Hz difference is more perceptually different than 10100-10200 Hz

4. **Output shape:** `(128 frequency bins, ~87 time steps)` for ~1 second of audio

### **Step 2: Log Conversion (Log-Mel-Spectrogram)**

```python
log_mel_spec = np.log(mel_spec + eps)  # eps=2.22e-16 (machine epsilon)
```

**Why logarithmic scale?**
- Human hearing perceives loudness logarithmically (not linearly)
- Compresses dynamic range (loud and quiet sounds both represented well)
- Improves model training by reducing outlier effects
- Typical values: dB scale [-40dB to 0dB] instead of [0 to 1000000]

### **Step 3: PCEN (Per-Channel Energy Normalization)**

```python
pcen = librosa.core.pcen(mel_spec, sr=22050)
```

**What PCEN does:**
- Applies **temporal smoothing** to each mel-channel independently
- Normalizes energy based on local temporal context
- More robust to loud background noise than log-mel
- Formula involves adaptive gain control per frequency bin
- **Output:** Similar shape to log-mel but with better noise robustness

**Key advantage:** PCEN is more robust than log-mel-spectrogram for noisy bioacoustic recordings where background noise varies over time.

---

## 4. FEATURE EXTRACTION CLASSES & METHODS

### **Feature_Extractor Class** 
**File:** [src/datamodules/components/pcen.py](baselines/dcase2024_task5/src/datamodules/components/pcen.py)

```python
class Feature_Extractor:
    """Extracts and normalizes audio features"""
    
    mean_std = {}  # Class variable storing normalization statistics
    
    def __init__(self, features, audio_path=[]):
        """
        Initialize with feature parameters
        Args:
            features: Config object with:
                - sr: 22050 (sample rate)
                - n_fft: 1024
                - hop_mel: 256
                - n_mels: 128
                - fmax: 11025
                - feature_types: "@"-separated suffix list (e.g., "pcen@mel")
            audio_path: List of directories to search for .wav files
        """
        self.sr = features.sr
        self.n_fft = features.n_fft
        self.hop = features.hop_mel
        self.n_mels = features.n_mels
        self.fmax = features.fmax
        self.feature_types = features.feature_types.split("@")
        
        # Recursively find all .wav files
        self.files = recursive_glob(audio_path, ".wav")
        
        # Compute mean/std for each feature type across ~1000 files
        self.update_mean_std()
    
    def update_mean_std(self):
        """
        Compute global mean/std for each feature type
        Used for feature normalization across entire dataset
        
        Process:
        1. For each feature type (e.g., "pcen", "mel"):
        2. Load pre-computed .npy files (e.g., audio_001_pcen.npy)
        3. Concatenate all features
        4. Store: Feature_Extractor.mean_std[type] = [mean, std]
        """
        for suffix in self.feature_types:
            features = []
            # Load ~1000 pre-computed feature files
            for audio_path in self.files[:1000]:
                feature_path = audio_path.replace(".wav", f"_{suffix}.npy")
                features.append(np.load(feature_path).flatten())
            
            all_data = np.concatenate(features)
            Feature_Extractor.mean_std[suffix] = [np.mean(all_data), np.std(all_data)]
    
    def extract_feature(self, audio_path, feature_types=None, normalized=True):
        """
        Extract and normalize features for a single audio file
        
        Args:
            audio_path: Path to .wav file
            feature_types: Which types to extract
            normalized: Apply z-score normalization?
        
        Returns:
            Array of shape (time_steps, n_feature_dims)
            where n_feature_dims = sum of all feature channel counts
        """
        features = []
        for suffix in (feature_types or self.feature_types):
            # Load pre-computed feature
            feature_path = audio_path.replace(".wav", f"_{suffix}.npy")
            feat = np.load(feature_path)
            
            if normalized:
                mean, std = Feature_Extractor.mean_std[suffix]
                feat = (feat - mean) / std  # Z-score normalization
            
            features.append(feat)
        
        # Concatenate across frequency dimension
        return np.concatenate(features, axis=1)
```

---

## 5. DATASET CLASSES & DYNAMIC FEATURE LOADING

### **PrototypeDynamicArrayDataSet Class**
**File:** [src/datamodules/components/dynamic_pcen_dataset.py](baselines/dcase2024_task5/src/datamodules/components/dynamic_pcen_dataset.py)

**Purpose:** Loads PCEN features dynamically during training (on-the-fly)

```python
class PrototypeDynamicArrayDataSet(Dataset):
    """
    Dynamically loads PCEN features from .npy files
    Performs random segmentation for data augmentation
    """
    
    def __init__(self, path, features, train_param):
        self.path = path
        self.features = features
        self.train_param = train_param
        
        # Initialize feature extractor
        self.fe = Feature_Extractor(
            features, 
            audio_path=[path.train_dir, path.eval_dir]
        )
        
        # Metadata dictionary structure:
        # meta[class_name] = {
        #     "info": [(start_sec, end_sec), ...],  # Positive segments
        #     "neg_info": [(start_sec, end_sec), ...],  # Negative segments
        #     "file": [audio_path1, audio_path2, ...],
        #     "duration": [duration1, duration2, ...]
        # }
        self.meta = {}
        self.pcen = {}  # Cache of loaded PCEN features
        
        # Parse CSV annotations and build metadata
        self.build_meta()
        self.classes = list(self.meta.keys())
    
    def __getitem__(self, idx):
        """
        Return a training example for episodic training
        
        Returns:
            - segment: PCEN features of shape (seg_len=17, n_mels=128)
            - class_idx: Integer class label
            - class_name: String class name
        """
        class_name = self.classes[idx]
        
        # Randomly select a positive example from this class
        segment = self.select_positive(class_name)
        
        # If using negative contrast learning
        if self.train_param.negative_train_contrast:
            segment_neg = self.select_negative(class_name)
            return (
                segment.astype(np.float32),
                segment_neg.astype(np.float32),
                self.classes2int[class_name] * 2,      # Positive class label
                self.classes2int[class_name] * 2 + 1,  # Negative class label
            )
        
        return segment.astype(np.float32), self.classes2int[class_name], class_name
    
    def select_positive(self, class_name):
        """
        Randomly select a positive example from annotated segments
        1. Pick random segment from class's positive annotations
        2. Extract sub-segment of fixed length
        3. Tile if too short
        """
        segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
        start, end = self.meta[class_name]["info"][segment_idx]
        
        segment = self.select_segment(
            start, end,
            self.pcen[self.meta[class_name]["file"][segment_idx]],
            seg_len=int(self.seg_len * self.fps)
        )
        return segment
    
    def select_segment(self, start, end, pcen, seg_len=17):
        """
        Extract fixed-length segment from PCEN features
        
        Args:
            start, end: Time boundaries in seconds
            pcen: Pre-loaded PCEN array (time_steps, 128)
            seg_len: Target segment length in frames
        
        Process:
        1. Convert time to frame indices
        2. If segment too short: tile (repeat) the segment
        3. If segment too long: random crop
        """
        start_frame = int(start * self.fps)
        end_frame = int(end * self.fps)
        duration_frames = end_frame - start_frame
        
        if duration_frames < seg_len:
            # Segment too short → tile/repeat until reaching seg_len
            x = pcen[start_frame:end_frame]
            tile_times = np.ceil(seg_len / duration_frames)
            x = np.tile(x, (int(tile_times), 1))
            x = x[:seg_len]  # Trim to exact size
        else:
            # Segment long enough → random crop
            rand_start = np.random.uniform(start_frame, end_frame - seg_len)
            x = pcen[int(rand_start):int(rand_start) + seg_len]
        
        return x
    
    def build_meta(self):
        """
        Parse all CSV annotation files and build metadata dictionary
        1. For each .csv file (contains onset/offset annotations)
        2. Extract class name and audio file path
        3. Build metadata with positive and negative segments
        """
        for csv_file in self.all_csv_files:
            glob_cls_name = self.get_glob_cls_name(csv_file)
            df_pos = self.get_df_pos(csv_file)  # Get positive rows
            start_time, end_time = self.get_time(df_pos)  # Parse onset/offset
            cls_list = self.get_cls_list(df_pos, glob_cls_name, start_time)
            
            # Update meta with these annotations
            self.update_meta(start_time, end_time, cls_list, csv_file)
```

---

## 6. FEATURE PARAMETERS FROM CONFIG

**File:** [configs/train.yaml](baselines/dcase2024_task5/configs/train.yaml) & [deep_learning/config.yaml](baselines/deep_learning/config.yaml)

```yaml
features:
  seg_len: 0.200          # Segment length in seconds (200ms)
  hop_seg: 0.100          # Hop between segments (100ms) - 50% overlap
  sr: 22050               # Sample rate (Hz)
  n_fft: 1024             # FFT size for STFT
  hop_mel: 256            # Hop length for mel-spectrogram
  n_mels: 128             # Number of mel-frequency bins
  fmax: 11025             # Maximum frequency (Hz) - Nyquist = sr/2
  fmin: 50                # Minimum frequency (Hz)

# Computed values:
# Time resolution (fps = frames per second)
#   fps = sr / hop_mel = 22050 / 256 ≈ 86.13 fps
# 
# Segment length in frames:
#   seg_len_frames = 0.200 * 86.13 ≈ 17 frames
#
# Therefore: each training example = (17 time steps, 128 mel bins)
```

---

## 7. TRAINING PIPELINE ARCHITECTURE

### **Data Flow During Training:**

```
┌─────────────────────────────────────────────────────────────────┐
│  PrototypeDataModule (PyTorch Lightning)                        │
│  - Loads PrototypeDynamicArrayDataSet                           │
│  - Uses IdentityBatchSampler for episodic batching              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  Episodic Batch [B, 17, 128]                                    │
│  - B = k_way * n_shot * 2 = 5 * 5 * 2 = 50                     │
│  - 5 classes (k_way), 5 support + 5 query per class            │
│  - 17 time steps, 128 mel-bins per sample                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  Encoder Network (ResNet / SimpleDenseNet)                      │
│  - Input: [50, 17, 128]                                         │
│  - Reshape to: [50, 1, 17, 128] (add channel dim for CNN)       │
│  - Process through conv blocks                                  │
│  - Output: [50, embedding_dim]                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  Prototypical Loss Computation                                  │
│  1. Compute prototypes (mean embedding per class)               │
│  2. Compute distances from query to prototypes                  │
│  3. Classify query examples                                     │
│  4. Compute loss and accuracy                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  Backward Pass & Optimization                                   │
│  - Update encoder weights                                       │
│  - Learning rate: 0.0001 with exponential decay (γ=0.5)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. MODEL ENCODER ARCHITECTURE

### **ResNet for Audio Features**
**File:** [baselines/deep_learning/Model.py](baselines/deep_learning/Model.py)

```python
class ResNet(nn.Module):
    """
    4-layer ResNet adapted for audio spectrograms
    Input: (batch, 1, time_steps, n_mels) = (B, 1, 17, 128)
    """
    
    def __init__(self, drop_rate=0.1):
        # 4 residual blocks with progressively larger channels
        self.layer1 = self._make_layer(BasicBlock, 64, stride=2)   # (B, 64, 8, 64)
        self.layer2 = self._make_layer(BasicBlock, 128, stride=2)  # (B, 128, 4, 32)
        self.layer3 = self._make_layer(BasicBlock, 64, stride=2)   # (B, 64, 2, 16)
        # layer4 is optional/not used
        
        self.pool = nn.AdaptiveAvgPool2d((4, 2))  # (B, 64, 4, 2) = 512 dims
    
    def forward(self, x):
        # x: (B, 17, 128)
        x = x.view(-1, 1, 17, 128)  # Add channel dimension
        
        x = self.layer1(x)  # Stride=2, spatial dims reduced
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)    # Global average pooling
        
        x = x.view(x.size(0), -1)  # Flatten: (B, 512)
        return x  # Return embedding vector
```

### **BasicBlock (Residual Block)**
```python
class BasicBlock(nn.Module):
    """
    Standard residual block with:
    - 3x3 convolutions
    - Batch normalization
    - Leaky ReLU activation
    - Skip connection
    - Max pooling
    """
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual  # Skip connection
        out = self.relu(out)
        out = self.maxpool(out)
        
        return out
```

---

## 9. PROTOTYPICAL LOSS FUNCTION

**File:** [src/utils/loss.py](baselines/dcase2024_task5/src/utils/loss.py)

The loss function is central to few-shot learning:

```python
def prototypical_loss(x_out, y, n_shot):
    """
    Prototypical Networks loss
    
    Args:
        x_out: Embeddings from encoder, shape [B, embedding_dim]
        y: Labels, shape [B]
        n_shot: Number of support examples per class (5)
    
    Process:
    1. Split batch into support (first n_shot * k_way samples)
       and query (remaining samples)
    2. Compute prototype = mean embedding per class from support set
    3. Compute distance matrix: query_embedding × class_prototypes
    4. Cross-entropy loss on distances
    5. Compute accuracy: classification correctness
    
    Output:
        loss: Scalar loss value
        acc: Classification accuracy
        supcon: Supplementary contrastive loss
    """
```

---

## 10. SUMMARY OF PREPROCESSING STEPS

### **From WAV to Training Input:**

1. **Load WAV file** → `librosa.load(path, sr=22050)` → audio signal
2. **Compute Mel-spectrogram** → 128 mel-frequency bins, ~86 fps frame rate
3. **Log scale conversion** → `log(mel_spec + eps)` → dynamic range compression
4. **Apply PCEN** → `librosa.core.pcen()` → noise-robust normalization
5. **Transpose** → Shape becomes `(time_steps, 128)` instead of `(128, time_steps)`
6. **Segment** → Extract `17 frame × 128 bin` patches from annotated regions
7. **Normalize** → Z-score: `(X - mean) / std` using dataset statistics
8. **Convert dtype** → `np.float32` for PyTorch compatibility
9. **Batch in episodes** → Group into episodic batches for meta-learning
10. **Feed to encoder** → ResNet processes and produces embeddings

### **Key Hyperparameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| sr | 22050 Hz | Sample rate |
| n_fft | 1024 | FFT window size |
| hop_mel | 256 | Hop size for spectrogram (11.6ms) |
| n_mels | 128 | Frequency bins |
| seg_len | 0.2 s | 17 frames × 11.6 ms |
| fps | 86.13 | Frames per second (sr/hop_mel) |
| n_shot | 5 | Support examples per class |
| k_way | 5 | Number of classes per episode |

---

## 11. KEY DIFFERENCES: PCEN vs LOG-MEL

| Aspect | Log-Mel-Spectrogram | PCEN |
|--------|-------------------|------|
| **Formula** | log(mel_spec) | Normalized log-mel with temporal smoothing |
| **Noise Robustness** | Moderate | High (temporal adaptation) |
| **Computation** | O(1) per frame | O(1) with state (temporal context) |
| **Use Case** | Speech, music | Bioacoustics, noisy environments |
| **Why used here** | Perceptual relevance | Better for bird/animal sounds with varying background |

---

## 12. ENTRY POINTS

### **Training:**
```bash
# Main entry point
python baselines/dcase2024_task5/train.py
# Uses: PrototypeDataModule → PrototypeDynamicArrayDataSet → PrototypeModule
```

### **Feature Extraction:**
```bash
# Older baseline using pre-computed h5 files
python baselines/deep_learning/main.py
# Uses: Feature_Extractor + h5py for storage
```

### **Testing/Inference:**
```bash
# Validation during training is integrated in PrototypeModule.test_step()
# Onset-offset detection implemented in test_step() method
```

---

## CONCLUSION

This codebase implements a **sophisticated few-shot learning system for bioacoustic detection**:

- **Preprocessing**: WAV → PCEN features (noise-robust, perceptually-motivated)
- **Data Loading**: Dynamic on-the-fly segmentation with class balancing
- **Model**: ResNet encoder + prototypical loss (metric learning)
- **Training**: Episodic meta-learning with support/query split
- **Output**: Embeddings that enable few-shot classification with minimal examples

The use of **PCEN** over standard log-mel-spectrogram is a key design choice for handling **real-world bioacoustic recordings** where background noise varies significantly.
