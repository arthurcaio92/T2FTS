"""This module provides examples of creating polling type-1 fuzzy sets.

PollingT1FuzzySet uses the variable "points" to store a dict of {x:mu} pairs.

To automatically generate a polling fuzzy set, a list of points can be given
to generate_fuzzy_sets.generate_polling_t1_fuzzy_set.
This creates the necessary dict, assigning membership values to each element.

The membership of an element is given as the percentage of its occurences.
The memberships may be normalised or not, set by gs.normalise_generated_sets.
"""

#import cPickle as pickle
import numpy as np
from decimal import Decimal
from collections import defaultdict

from fuzzycreator.fuzzy_sets.polling_t1_fuzzy_set import PollingT1FuzzySet
from fuzzycreator.measures import similarity_t1
from fuzzycreator.measures import distance_t1
from fuzzycreator import generate_fuzzy_sets
from fuzzycreator import global_settings as gs
from fuzzycreator import visualisations


gs.global_uod = [0, 10]
gs.global_x_disc = 1001
gs.normalise_generated_sets = False


def create_fuzzy_set_from_data(d):
    """Plot fuzzy sets generated from data given by load_data()."""
    fuzzy_sets = [generate_fuzzy_sets.generate_polling_t1_fuzzy_set(d)]
    visualisations.plot_sets(fuzzy_sets)


def create_t2_fuzzy_set_from_data(d):
    """Plot type-2 fuzzy sets generated from data given by load_data().

    The data is split into three subsets.
    Each subset is constructed into a type-1 fuzzy set.
    The three type-1 sets are then aggregated into a type-2 fuzzy set.
    """
    print ('Building type-2 fuzzy set')
    fs = generate_fuzzy_sets.generate_polling_t2_fuzzy_set(d)
    print ('Building 3-dimensional plot')
    fs.plot_set_3d()
    print ('Building 2-dimensional plot')
    fs.plot_set()


def bimodal_distribution_example():
    """Show polling and Gaussian fuzzy sets based on bi-modal data."""
    d1 = list(np.random.normal(2, 1, 1000))
    d2 = list(np.random.normal(7, 1, 1000))
    d1.extend(d2)
    d1 = [round(i, 1) for i in d1]
    fs = generate_fuzzy_sets.generate_polling_t1_fuzzy_set(d1)
    fs.plot_set()
    fs = generate_fuzzy_sets.generate_gaussian_t1_fuzzy_set(d1)
    fs.plot_set()

if __name__ == '__main__':
    d1 = list(np.random.normal(5, 1, 1000))
    d2 = list(np.random.normal(5, 0.5, 1000))
    d3 = list(np.random.normal(5, 0.25, 1000))
    d1 = [round(i, 1) for i in d1]
    d2 = [round(i, 1) for i in d2]
    d3 = [round(i, 1) for i in d3]
    create_fuzzy_set_from_data(d1)
    create_t2_fuzzy_set_from_data([d1, d2, d3])
    print ('Demonstrating bimodal distributions.')
    #bimodal_distribution_example()
