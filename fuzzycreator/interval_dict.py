"""This module is used to create an interval-based dict."""


class IntervalDict(object):
    """This class stores a dict in which the keys are intervals."""

    def __init__(self, overwrite_with_max=True):
        """Initiate the interval dict.

        When overwrite_with_max = True:
        If a key has been assigned multiple values the max is returned.
        When overwrite_with_max = False:
        If a key has been assigned multiple values the sum is returned.
        """
        # store a list of [(interval, value)] mappings
        self._interval_value_pairs = []
        self.overwrite_with_max = overwrite_with_max

    def __setitem__(self, _slice, _value):
        """Assign _value to the continuous _slice.

        _slice must be a slice; e.g. [1:3]
        _value must be numerical
        """
        if not isinstance(_slice, slice):
            raise Exception('The key must be a slice object.')
        if _slice.start is None or _slice.stop is None:
            raise Exception('Both end points must be given')
        self._interval_value_pairs.append(([_slice.start, _slice.stop],
                                           _value))

    def __getitem__(self, _point):
        """Return the value of the singleton _point."""
        values = [0]
        for interval, value in self._interval_value_pairs:
            if self._is_within_interval(interval, _point):
                values.append(value)
        if self.overwrite_with_max:
            return max(values)
        else:
            return sum(values)

    def singleton_keys(self):
        """Return the list of key values as singletons."""
        item_list = []
        for interval, value in self._interval_value_pairs:
            if interval[0] not in item_list:
                item_list.append(interval[0])
            if interval[1] not in item_list:
                item_list.append(interval[1])
        return item_list

    def keys(self):
        """Return the list of intervals used as keys."""
        item_list = []
        for interval, value in self._interval_value_pairs:
            item_list.append(interval)
        return item_list

    def values(self):
        """Return the list of values stored."""
        value_list = []
        for interval, value in self._interval_value_pairs:
            value_list.append(value)
        return value_list

    def _is_within_interval(self, interval, point):
        """Check if point is within the given interval."""
        if isinstance(interval[0], tuple):
            for s in interval:
                if point >= s[0] and point <= s[1]:
                    return True
        else:
            if point >= interval[0] and point <= interval[1]:
                return True
        return False
