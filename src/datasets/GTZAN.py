import librosa
import numpy as np
import music_fsl.util as util

from collections import defaultdict
from typing import BinaryIO, Optional, TextIO, Tuple, List, Dict
from deprecated.sphinx import deprecated
from mirdata import download_utils, jams_utils, core, io, annotations
from music_fsl.data import ClassConditionalDataset

BIBTEX = """@article{tzanetakis2002gtzan,
  title={GTZAN genre collection},
  author={Tzanetakis, George and Cook, P},
  journal={Music Analysis, Retrieval and Synthesis for Audio Signals},
  year={2002}
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(
        filename="gtzan_genre_index_1.0.json",
        partial_download=["all", "tempo_beat_annotations"],
    ),
    "mini": core.Index(
        filename="gtzan_genre_1.0_mini_index.json",
        partial_download=["mini", "tempo_beat_annotations"],
    ),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="genres.tar.gz",
        url="https://www.dropbox.com/scl/fi/8pgz2qr23w2gjj8krudn8/gtanzgenres.tar.gz?rlkey=plhy147pfgz1o2rwcifyk592i&st=34imx3ao&dl=1",
        checksum="5c27114dad84e9284c52b2827633f692",
        destination_dir="gtzan_genre",
    ),
    "mini": download_utils.RemoteFileMetadata(
        filename="main.zip",
        url="https://github.com/TempoBeatDownbeat/gtzan_mini/archive/refs/heads/main.zip",
        checksum="44f7f23af8363d96c59663a987f29a4c",
    ),
    "tempo_beat_annotations": download_utils.RemoteFileMetadata(
        filename="annot.zip",
        url="https://github.com/TempoBeatDownbeat/gtzan_tempo_beat/archive/refs/heads/main.zip",
        checksum="4baa58112697a8087de04558d6e97442",
    ),
}

LICENSE_INFO = "Unfortunately we couldn't find the license information for the GTZAN_genre dataset."


class Track(core.Track):
    """gtzan_genre Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the audio file
        genre (str): annotated genre
        track_id (str): track id

    Cached Properties:
        beats (BeatData): human-labeled beat annotations
        tempo (float): global tempo annotations

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.genre = track_id.split(".")[0]
        if self.genre == "hiphop":
            self.genre = "hip-hop"

        self.audio_path = self.get_path("audio")
        self.beats_path = self.get_path("beats")
        self.tempo_path = self.get_path("tempo")

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @core.cached_property
    def tempo(self) -> Optional[float]:
        return load_tempo(self.tempo_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            tags_gtzan_data=[(self.genre, "gtzan-genre")],
            beat_data=[(self.beats, None)],
            tempo_data=[(self.tempo, None)],
            metadata={
                "title": "Unknown track",
                "artist": "Unknown artist",
                "release": "Unknown album",
                "duration": 30.0,
                "curator": "George Tzanetakis",
            },
        )


@io.coerce_to_string_io
def load_beats(fhandle: TextIO) -> annotations.BeatData:
    """Load GTZAN format beat data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a beat annotation file

    Returns:
        BeatData: loaded beat data

    """
    beats = np.loadtxt(fhandle, ndmin=2)
    times = beats[:, 0]
    try:
        positions = beats[:, 1]
    except IndexError:
        positions = None
    beat_data = annotations.BeatData(
        times=times, time_unit="s", positions=positions, position_unit="bar_index"
    )

    return beat_data


@io.coerce_to_string_io
def load_tempo(fhandle: TextIO) -> float:
    """Load GTZAN format tempo data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a beat annotation file

    Returns:
        tempo (float): loaded tempo data

    """

    tempo = np.loadtxt(fhandle, ndmin=2)

    return float(tempo)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a GTZAN audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=22050, mono=True)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class GTZANDataset(core.Dataset):
    """
    The gtzan_genre dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="gtzan_genre",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.gtzan_genre.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)




class GTZAN(ClassConditionalDataset):
    """
    Initialize a `GTZAN` dataset instance.

    Args:
        classes (List[str]): A list of genra classes to include in the dataset.
        duration (float): The duration of each audio clip in the dataset (in seconds).
        sample_rate (int): The sample rate of the audio clips in the dataset (in Hz).
    """

    TRAIN_CLASSES = [
        'classical', 'country', 'disco', 'hip-hop', 'jazz', 'rock'
    ]

    TEST_CLASSES = [
        'blues', 'reggae', 'pop', 'metal'
    ]

    GENRES = [
        'classical', 'country', 'disco', 'hip-hop', 'jazz', 'rock', 'blues', 'reggae', 'pop', 'metal'
    ]

    def __init__(self,
            classes: List[str] = None,
            duration: float = 30.0,
            sample_rate: int = 16000,
            dataset_path: str = None
        ):
        if classes is None:
            classes = self.GENRES

        self.classes = classes
        self.duration = duration
        self.sample_rate = sample_rate

        # initialize the tinysol dataset and download if necessary
        self.dataset = GTZANDataset()
        self.dataset.download()

        # make sure the instruments passed in are valid
        for genra in classes:
            assert genra in self.GENRES, f"{genra} is not a valid genra"

        # load all tracks for this instrument
        self.tracks = []
        for track in self.dataset.load_tracks().values():
            # look for invalid file
            if track.audio_path.split('/')[-1] !='jazz.00054.wav':
              if track.genre in self.classes:
                self.tracks.append(track)


    @property
    def classlist(self) -> List[str]:
        return self.classes

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        if not hasattr(self, "_class_to_indices"):
            self._class_to_indices = defaultdict(list)
            for i, track in enumerate(self.tracks):
                self._class_to_indices[track.genre].append(i)

        return self._class_to_indices

    def __getitem__(self, index) -> Dict:
        # load the track for this index
        track = self.tracks[index]

        # load the excerpt
        data = util.load_excerpt(track.audio_path, self.duration, self.sample_rate)
        data["label"] = track.genre

        return data

    def __len__(self) -> int:
        return len(self.tracks)