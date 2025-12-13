Model: Prototypical Network (ProtoNet)
Embedding dim: 128
Distance: Euclidean

Training setup:
- Episodes per epoch: 1000
- Epochs: 10
- Total episodes: 10,000
- k-way: 2
- n-shot: 5
- n-query: 3
- Batch size: 1 episode
- Optimizer: Adam
- Learning rate: 1e-3
- Device: CUDA

Dataset:
- DCASE-style Few-shot Bioacoustic Event Detection
- Training_Set
- Validation_Set_DSAI_2025_2026

Date trained: <13-12-2025>
Notes:
- Reduced training budget due to GPU limits
