from .AB import ABLoss
from .AT import Attention
from .AutoKD import AutoKD, Batch_kl, Channel_gmml2, Channel_kl, Spatial_kl
from .CC import Correlation
from .FitNet import HintLoss
from .FSP import FSP
from .FT import FactorTransfer
from .ICKD import ICKDLoss
from .KD import DistillKL
from .KDSVD import KDSVD
from .NST import NSTLoss
from .PKT import PKT
from .RKD import RKDLoss
from .RMI import RMIloss
from .SP import Similarity
from .VID import VIDLoss


def build_loss(name: str = None, **kwargs):
    if name == 'AB':
        return ABLoss()
    elif name == 'AT':
        return Attention()
    elif name == 'CC':
        return Correlation()
    elif name == 'FitNet':
        return HintLoss()
    elif name == 'FSP':
        return FSP()
    elif name == 'ICKD':
        return ICKDLoss()
    elif name == 'FT':
        return FactorTransfer()
    elif name == 'KD':
        return DistillKL()
    elif name == 'KDSVD':
        return KDSVD()
    elif name == 'NST':
        return NSTLoss()
    elif name == 'PKT':
        return PKT()
    elif name == 'RKD':
        return RKDLoss()
    elif name == 'SP':
        return Similarity()
    elif name == 'VID':
        return VIDLoss()
    else:
        raise f'Not support type: {name}'
