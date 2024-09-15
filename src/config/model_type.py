from enum import Enum

from models.FewShotLearnerUnlabeled import FewShotLearnerUnlabeled
from models.PrototypicalNetKSoft import PrototypicalNetKSoft
from models.PrototypicalNetKSoftWithDistractor import PrototypicalNetKSoftWithDistractor
from models.PrototypicalNetKSoftWithDistractorMaskModel import PrototypicalNetKSoftWithDistractorMaskModel
from models.NegPrototypicalNet import NegPrototypicalNet
from models.MUSIC import MUSIC
from models.MUSICWithDistractor import MUSICWithDistractor


class PrototypicalNetType(Enum):
    softkmeans = PrototypicalNetKSoft
    softkmeansdistractor = PrototypicalNetKSoftWithDistractor
    softkmeansmasked = PrototypicalNetKSoftWithDistractorMaskModel
    music = NegPrototypicalNet
    musicdistractor = NegPrototypicalNet

class FewShotLearnerType(Enum):
    kmeans = FewShotLearnerUnlabeled
    music = MUSIC
    musicdistractor = MUSICWithDistractor
