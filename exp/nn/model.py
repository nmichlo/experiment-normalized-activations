from typing import List

import numpy as np


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def get_ae_layer_sizes(start, mid, r=1) -> List[int]:
    assert r >= 0
    down = np.linspace(start, mid, r + 1)[:-1]
    return np.array([*down, mid, *down[::-1]], dtype='int').tolist()


def get_layer_sizes(start, end, r=1) -> List[int]:
    assert r >= 0
    return np.array(np.linspace(start, end, r * 2 + 1), dtype='int').tolist()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
