"""This module lists fuzzy set based exceptions.

These are regarding non-existent parts of fuzzy sets,
e.g. empty alpha-cuts, empty zLevels.
"""


class AlphaCutError(Exception):
    """The alpha-cut exceeds the height of the fuzzy set."""

    pass


class ZLevelError(Exception):
    """The zlevel exceeds the secondary height of the fuzzy set."""

    pass
