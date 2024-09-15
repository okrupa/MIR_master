from enum import Enum

from datasets.TinySOL import TinySOL
from datasets.GTZAN import GTZAN
from datasets.CustomDataset import CustomDataset

class DatasetType(Enum):
    tiny_sol = TinySOL
    gtzan = GTZAN
    custom = CustomDataset

