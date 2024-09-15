import mirdata

from collections import defaultdict
from typing import List, Dict
import music_fsl.util as util

from music_fsl.data import ClassConditionalDataset


class TinySOL(ClassConditionalDataset):
    """
    Initialize a `TinySOL` dataset instance.

    Args:
        classes (List[str]): A list of instrument classes to include in the dataset.
        duration (float): The duration of each audio clip in the dataset (in seconds).
        sample_rate (int): The sample rate of the audio clips in the dataset (in Hz).
    """

    TRAIN_CLASSES = [
        'French Horn',
        'Violin',
        'Flute',
        'Contrabass',
        'Trombone',
        'Cello',
        'Clarinet in Bb',
        'Oboe',
        'Accordion'
    ]

    TEST_CLASSES = [
        'Bassoon',
        'Viola',
        'Trumpet in C',
        'Bass Tuba',
        'Alto Saxophone'
    ]

    INSTRUMENTS = [
        'Bassoon', 'Viola', 'Trumpet in C', 'Bass Tuba',
        'Alto Saxophone', 'French Horn', 'Violin',
        'Flute', 'Contrabass', 'Trombone', 'Cello',
        'Clarinet in Bb', 'Oboe', 'Accordion'
    ]

    def __init__(self,
            classes: List[str] = None,
            duration: float = 1.0,
            sample_rate: int = 16000,
            dataset_path: str = None
        ):
        if classes is None:
            classes = self.INSTRUMENTS

        self.classes = classes
        self.duration = duration
        self.sample_rate = sample_rate

        self.dataset = mirdata.initialize('tinysol')
        self.dataset.download()

        for instrument in classes:
            assert instrument in self.INSTRUMENTS, f"{instrument} is not a valid instrument"

        self.tracks = []
        for track in self.dataset.load_tracks().values():
            if track.instrument_full in self.classes:
                self.tracks.append(track)


    @property
    def classlist(self) -> List[str]:
        return self.classes

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        if not hasattr(self, "_class_to_indices"):
            self._class_to_indices = defaultdict(list)
            for i, track in enumerate(self.tracks):
                self._class_to_indices[track.instrument_full].append(i)

        return self._class_to_indices

    def __getitem__(self, index) -> Dict:
        track = self.tracks[index]

        data = util.load_excerpt(track.audio_path, self.duration, self.sample_rate)
        data["label"] = track.instrument_full

        return data

    def __len__(self) -> int:
        return len(self.tracks)