import os
import librosa
import numpy as np
import music_fsl.util as util
import torch
import random
import music_fsl.util as util

from collections import defaultdict
from typing import List, Dict
from music_fsl.data import ClassConditionalDataset


class PredictDataset(ClassConditionalDataset):
    """
    Initialize a `Predict Data Dataset Loader` dataset instance.

    """

    def __init__(self, 
            duration: float = 1.0, 
            sample_rate: int = 16000,
            dataset_path: str = None
        ):

        self.duration = duration
        self.sample_rate = sample_rate
        self.dataset_path = dataset_path

        # load all tracks for this instrument
        self.tracks = []
        for subdir_dir, dirs_dir, files_dir in os.walk(self.dataset_path):
            for file in files_dir:
                if file.endswith('.wav'):
                    if librosa.get_duration(filename=os.path.join(self.dataset_path, file)) >= duration:
                        self.tracks.append(os.path.join(self.dataset_path, file))

    def __getitem__(self, index) -> Dict:
        # load the track for this index
        track = self.tracks[index]

        # load the excerpt
        data = util.load_excerpt(track, self.duration, self.sample_rate)

        data["name"] = track

        return data

    def __len__(self) -> int:
        return len(self.tracks)


class PredictDatasetLoader(torch.utils.data.Dataset):
    def __init__(self,
        dataset: ClassConditionalDataset,
    ):
        self.dataset = dataset
    
    def __getitem__(self, index: int) -> Dict:
        n_tracks = len(self.dataset)
        data = []
        for i in range(n_tracks):
            data.append(self.dataset[i])
        
        data = util.collate_list_of_dicts(data)
        return data
