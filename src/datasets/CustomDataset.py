import os
import mirdata
import librosa
import torch
import random
import music_fsl.util as util

from music_fsl.data import ClassConditionalDataset
from collections import defaultdict
from typing import List, Dict, Tuple


class CustomDataset(ClassConditionalDataset):

    def __init__(self, 
            classes: List[str] = None,
            duration: float = 1.0, 
            sample_rate: int = 16000,
            dataset_path: str = None
        ):
        self.classes = classes  
        self.duration = duration
        self.sample_rate = sample_rate
        self.dataset_path = dataset_path

        self.tracks = []
        self.tracks_unlabeled = []
        for dir in os.listdir(os.path.join(self.dataset_path, 'labeled')):
            if dir in self.classes:
                for subdir_dir, dirs_dir, files_dir in os.walk(os.path.join(self.dataset_path, 'labeled', dir)):
                    for file in files_dir:
                        if file.endswith('.wav'):
                            if librosa.get_duration(filename=os.path.join(self.dataset_path, 'labeled', dir, file)) >= self.duration:
                                self.tracks.append([os.path.join(self.dataset_path, 'labeled', dir, file), dir])

        for subdir_dir, dirs_dir, files_dir in os.walk(os.path.join(self.dataset_path, 'unlabeled')):
            for file in files_dir:
                if file.endswith('.wav'):
                    if librosa.get_duration(filename=os.path.join(self.dataset_path, 'unlabeled', file)) >= self.duration:
                        self.tracks_unlabeled.append([os.path.join(self.dataset_path, 'unlabeled', file)])


    @property
    def classlist(self) -> List[str]:
        return self.classes

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        # cache it in self._class_to_indices 
        # so we don't have to recompute it every time
        if not hasattr(self, "_class_to_indices"):
            self._class_to_indices = defaultdict(list)
            for i, track in enumerate(self.tracks):
                self._class_to_indices[track[1]].append(i)

        return self._class_to_indices

    def __getitem__(self, index) -> Dict:
        # load the track for this index
        track = self.tracks[index]

        # load the excerpt
        data = util.load_excerpt(track[0], self.duration, self.sample_rate)
        data["label"] = track[1]

        return data

    def get_unlabeled(self, index) -> Dict:
        # load the track for this index
        track = self.tracks_unlabeled[index]

        # load the excerpt
        data = util.load_excerpt(track[0], self.duration, self.sample_rate)
        data["label"] = track[0]

        return data

    def __len__(self) -> int:
        return len(self.tracks)


class CustomDatasetLoader(torch.utils.data.Dataset):
    """
        A dataset for sampling few-shot learning tasks from a class-conditional dataset with unlabeled data.

    Args:
        dataset (ClassConditionalDataset): The dataset to sample episodes from.
        n_way (int): The number of classes to sample per episode.
            Default: 5.
        n_support (int): The number of samples per class to use as support.
            Default: 5.
        n_query (int): The number of samples per class to use as query.
            Default: 20.
        n_unlabeled (int): The number of samples per class to use as unlabeled data.
            Default: 5.
    """
    def __init__(self,
        dataset: ClassConditionalDataset,
        n_way: int = 5,
        n_support: int = 5,
        n_query: int = 20,
        n_unlabeled: int = 5,
        n_episodes: int = 100,
    ):
        self.dataset = dataset

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_unlabeled = n_unlabeled
        self.n_episodes = n_episodes

    def __getitem__(self, index: int) -> Tuple[Dict, Dict, Dict]:
        """Sample an episode from the class-conditional dataset.

        Each episode is a tuple of three dictionaries: a support set , a unlabeled set and a query set.
        The support set contains a set of samples from each of the classes in the
        episode, the unlabeled set contains a set of samples from each of the classes (optionaly with classes from distractor) in the
        episode without labels and the query set contains another set of samples from each of the
        classes. The class labels are added to each item in the support and query
        sets, and the list of classes is also included in each dictionary.

        Yields:
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: A tuple containing the support
            set, unlabeled set and the query set for an episode.
        """
        rng = random.Random(index)

        episode_classlist = rng.sample(self.dataset.classlist, self.n_way)

        support, unlabeled, query = [], [], []
        for c in episode_classlist:
            all_indices = self.dataset.class_to_indices[c]

            try:
                indices = rng.sample(all_indices, self.n_support + self.n_query)
            except:
                raise Exception("Dataset to small, change config ore upload more files")
            # indices = rng.sample(all_indices, self.n_support + self.n_query)
            items = [self.dataset[i] for i in indices]

            for item in items:
                item["target"] = torch.tensor(episode_classlist.index(c))

            support.extend(items[:self.n_support])
            query.extend(items[self.n_support:])

        if len(self.dataset.tracks_unlabeled) < self.n_unlabeled:
            raise Exception("Unlabeled data to small")

        indices = random.sample(range(len(self.dataset.tracks_unlabeled)), self.n_unlabeled)
        for idx in indices:
            data = self.dataset.get_unlabeled(idx)
            unlabeled.append(data)


        # collate the support and query sets
        support = util.collate_list_of_dicts(support)
        unlabeled = util.collate_list_of_dicts(unlabeled)
        query = util.collate_list_of_dicts(query)

        support["classlist"] = episode_classlist
        query["classlist"] = episode_classlist

        return support, unlabeled, query

    def __len__(self):
        return self.n_episodes
