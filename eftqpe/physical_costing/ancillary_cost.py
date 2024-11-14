import numpy as np

from qualtran.surface_code import MagicCount
from qualtran.bloqs.qft.qft_phase_gradient import QFTPhaseGradient
from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState

def qft_cost(dimension: int) -> MagicCount:
    """Toffoli cost of the QFT. 
    
    We approximate this by picking the first largest power of 2 as register dimension.
    TODO: implement QFT bloq for arbitrary control dimensions

    Args:
        dimension: dimension of the QPE control register (T_max + 1)
    """
    bitsize = int(np.ceil(np.log2(dimension)))
    qft = QFTPhaseGradient(bitsize)

    # in phase gradient QFT, all T gates are used for implementing Toffolis
    cost = MagicCount(n_ccz = qft.t_complexity.t / 4)

    return cost


def sin_state_cost(dimension: int) -> MagicCount:
    """Toffoli cost of preparing the sine state. 
    
    We approximate this by picking the first largest power of 2 as register dimension.
    TODO: implement sine state preparation bloq for arbitrary control dimensions

    Args:
        dimension: dimension of the QPE control register (T_max + 1)
    """
    bitsize = int(np.ceil(np.log2(dimension)))
    state_prep = LPResourceState(bitsize)

    # in phase gradient QFT, all T gates are used for implementing Toffolis
    cost = MagicCount(n_ccz = state_prep.t_complexity.t / 4)

    return cost


def ancillary_cost(dimension: int) -> MagicCount:
    """Toffoli cost of ancillary operations (sine state preparation and QFT).

    We approximate this by picking the first largest power of 2 as register dimension.

    Args:
        dimension: dimension of the QPE control register (T_max + 1)
    """

    return qft_cost(dimension) + sin_state_cost(dimension)
