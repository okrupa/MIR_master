import torch
import random
import music_fsl.util as util

from typing import Tuple, Dict


from music_fsl.data import ClassConditionalDataset


class EpisodeDatasetUnlabeled(torch.utils.data.Dataset):
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
        n_distractor (int): Number of distractor classes.
            Default: 0.
        n_episodes (int): The number of episodes to generate.
            Default: 100.
    """
    def __init__(self,
        dataset: ClassConditionalDataset,
        n_way: int = 5,
        n_support: int = 5,
        n_query: int = 20,
        n_unlabeled: int = 5,
        n_distractor: int = 0,
        n_episodes: int = 100,
    ):
        self.dataset = dataset

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_unlabeled = n_unlabeled
        self.n_distractor = n_distractor
        self.n_episodes = n_episodes

    def __getitem__(self, index: int) -> Tuple[Dict, Dict]:
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

            indices = rng.sample(all_indices, self.n_support + self.n_unlabeled + self.n_query)
            items = [self.dataset[i] for i in indices]

            for item in items:
                item["target"] = torch.tensor(episode_classlist.index(c))

            support.extend(items[:self.n_support])
            unlabeled.extend(items[self.n_support:(self.n_support + self.n_unlabeled)])
            query.extend(items[(self.n_support + self.n_unlabeled):])


        # collate the support and query sets
        support = util.collate_list_of_dicts(support)
        unlabeled = util.collate_list_of_dicts(unlabeled)
        query = util.collate_list_of_dicts(query)

        support["classlist"] = episode_classlist
        query["classlist"] = episode_classlist

        return support, unlabeled, query

    def __len__(self):
        return self.n_episodes

    def print_episode(self, support, ulabeled, query):
        """Print a summary of the support, ulabeled and query sets for an episode.

        Args:
            support (Dict[str, Any]): The support set for an episode.
            ulabeled (Dict[str, Any]): The ulabeled set for an episode.
            query (Dict[str, Any]): The query set for an episode.
        """
        print("Support Set:")
        print(f"  Classlist: {support['classlist']}")
        print(f"  Audio Shape: {support['audio'].shape}")
        print(f"  Target Shape: {support['target'].shape}")
        print()
        print("Ulabeled Set:")
        print(f"  Audio Shape: {ulabeled['audio'].shape}")
        print(f"  Target Shape: {ulabeled['target'].shape}")
        print()
        print("Query Set:")
        print(f"  Classlist: {query['classlist']}")
        print(f"  Audio Shape: {query['audio'].shape}")
        print(f"  Target Shape: {query['target'].shape}")
