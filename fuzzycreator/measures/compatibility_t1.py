"""This module contains a compatibility measure for type-1 fuzzy sets.

See http://ieeexplore.ieee.org/abstract/document/6891672/ for further details.
"""


from decimal import Decimal

from . import similarity_t1
from . import distance_t1
from .. import global_settings as gs


def compatibility(fs1, fs2, w0=0.7, w1=0.3):
    """Calculate weighted average of dissimilarity and directional distance.

    w0 and w1 are the weights given to dissimilarity and distance, resp.
    Result is in the range [-1, 1].
    0 is for identical sets and -1 and 1 are maximum distance.
    """
    w0 = Decimal(str(w0))
    w1 = Decimal(str(w1))
    similarity = similarity_t1.jaccard(fs1, fs2)
    distance = distance_t1.mcculloch(fs1, fs2)
    max_distance = max(fs1.uod[1], fs2.uod[1]) - min(fs1.uod[0], fs1.uod[0])
    distance = distance / max_distance
    dissimilarity = 1 - similarity
    # make both have same sign
    if distance < 0:
        dissimilarity = -dissimilarity
    return gs.rnd((w0 * dissimilarity) + (w1 * distance))
