import numpy as np
import os
from glob import glob
from tqdm import tqdm
from pathlib import Path


def recursive_glob(
    path: Path, 
    suffix: str
):
    return (
        glob(os.path.join(path, "*" + suffix))
        + glob(os.path.join(path, "*/*" + suffix))
        + glob(os.path.join(path, "*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*/*/*" + suffix))
    )


class Feature_Extractor:
    mean_std = {}

    def __init__(self, features, audio_path=None, stats_audio_path=None):
        self.sr = features.sr
        self.n_fft = features.n_fft
        self.hop = features.hop_mel
        self.n_mels = features.n_mels
        self.fmax = features.fmax
        self.feature_types = features.feature_types.split("@")
        if audio_path is None:
            audio_path = []
        if stats_audio_path is None:
            stats_audio_path = audio_path
        self.files = []
        self.stats_key = tuple(
            sorted(
                os.path.abspath(p)
                for p in stats_audio_path
                if p is not None
            )
        )
        # stats_audio_path is the sole source for normalization statistics.
        for each in stats_audio_path:
            if each is not None:
                assert os.path.exists(each), "Path not found: %s" % each
            print("Looking for data in: ", os.path.abspath(each))
            data = list(recursive_glob(each, ".wav"))
            self.files += data
            print("Find %s audio files" % (len(data)))

        self.files = sorted(self.files)
        self.update_mean_std()
        self.feature_lens = []

    def update_mean_std(self, feature_types=None):
        if not self.stats_key:
            raise RuntimeError(
                "No stats_audio_path provided for normalization. "
                "Pass stats_audio_path pointing to the training set."
            )
        if not self.files:
            raise RuntimeError(
                "No audio files found for mean/std computation. "
                "Pass stats_audio_path pointing to the training set."
            )
        print("Calculating mean and std")
        for suffix in self.feature_types if (feature_types is None) else feature_types:
            cache_key = (self.stats_key, suffix)
            if cache_key in Feature_Extractor.mean_std:
                continue
            print("Calculating: ", suffix)
            features = []
            for audio_path in tqdm(self.files[:1000]):
                feature_path = audio_path.replace(".wav", "_%s.npy" % suffix)
                features.append(np.load(feature_path).flatten())
            all_data = np.concatenate(features)
            Feature_Extractor.mean_std[cache_key] = [
                np.mean(all_data),
                np.std(all_data),
            ]
        print(Feature_Extractor.mean_std)

    def _ensure_time_major(self, feature: np.ndarray) -> np.ndarray:
        if feature.ndim != 2:
            return feature
        # Older exports used (n_mels, n_frames); datasets expect (n_frames, n_mels).
        if feature.shape[0] == self.n_mels and feature.shape[1] != self.n_mels:
            return feature.T
        return feature

    def extract_feature(self, audio_path, feature_types=None, normalized=True):
        features = []
        for suffix in self.feature_types if (feature_types is None) else feature_types:
            feature_path = audio_path.replace(".wav", "_%s.npy" % suffix)
            if not normalized:
                loaded = np.load(feature_path)
            else:
                mean, std = Feature_Extractor.mean_std[(self.stats_key, suffix)]
                loaded = (np.load(feature_path) - mean) / std
            loaded = self._ensure_time_major(loaded)
            features.append(loaded)
            self.feature_lens.append(features[-1].shape[1])
        return np.concatenate(features, axis=1)

    # def apply_mask(self, features, mask):
    #     start = 0
    #     import ipdb; ipdb.set_trace()
    #     for len, suffix in zip(self.feature_lens, self.feature_types):
    #         if(len != 128):
    #             continue
    #         else:
    #             mean, std = Feature_Extractor.mean_std[suffix]
    #             features[:, start:start +len] = (features[:, start:start +len] * std + mean) * mask
    #             features[:, start:start +len] = (features[:, start:start +len] - mean) / std
    #         start += len
