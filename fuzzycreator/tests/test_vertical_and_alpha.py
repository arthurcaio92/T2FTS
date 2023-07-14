"""This module is to visually test that membership and alpha-cuts match."""

import matplotlib.pyplot as plt
from numpy import linspace
from operator import itemgetter
from collections import defaultdict

from fuzzycreator.membership_functions.trapezoidal import Trapezoidal
from fuzzycreator.membership_functions.triangular import Triangular
from fuzzycreator.membership_functions.gaussian import Gaussian
from fuzzycreator.fuzzy_sets.polling_t1_fuzzy_set import PollingT1FuzzySet
from fuzzycreator.fuzzy_sets.fuzzy_set import FuzzySet
import fuzzycreator.global_settings as gs
from fuzzycreator.fuzzy_exceptions import *


def plot_type1_set_by_vertical_and_alpha(fs):
    """Create plots of the fuzzy set.

    Use both alpha-cuts (black) and vertical slices (red).
    Both approaches should look the same so it is a useful
    way of ensuring the alpha-cut method is correct.
    """
    fig = plt.figure()
    axes = fig.add_subplot(111)
    # plot vertical slices
    plot_points = [round(x, 2) for x in
                   linspace(fs.uod[0]-1, fs.uod[1]+1, gs.global_x_disc)]
    plt.plot(plot_points,
             [fs.calculate_membership(x) for x in plot_points],
             color='#FF0000', linewidth=3)
    # plot alpha cuts
    TICKS = (fs.uod[1] - fs.uod[0]) + 1
    points = defaultdict(int)
    for y in linspace(0, 1.0, gs.global_alpha_disc):
        try:
            cut = fs.calculate_alpha_cut(y)
            # Flatten the lists if it's non-congvex
            if isinstance(cut[0], tuple):
                cut = [val for sublist in cut for val in sublist]
            for x in cut:
                points[x] = max(points[x], round(y, 4))
        except AlphaCutError:
            pass
    X = sorted(points.keys())
    Y = [points[x] for x in X]
    plt.plot(X, Y, linewidth=3, color='#000000')
    axes.set_ylim(0, 1.01)
    # increase the boundary to make sure it doesn't fall outside of it
    axes.set_xlim(fs.uod[0]-2, fs.uod[1]+2)
    plt.show()


fs = FuzzySet(Triangular(2, 3, 4))
#plot_type1_set_by_vertical_and_alpha(fs)
