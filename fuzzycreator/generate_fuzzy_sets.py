"""This module is used to generate fuzzy sets from data."""

from collections import Counter
from decimal import Decimal
from numpy import mean, std

from .fuzzy_sets.fuzzy_set import FuzzySet
from .fuzzy_sets.discrete_t1_fuzzy_set import DiscreteT1FuzzySet
from .fuzzy_sets.polling_t1_fuzzy_set import PollingT1FuzzySet
from .fuzzy_sets.t2_aggregated_fuzzy_set import T2AggregatedFuzzySet
from .membership_functions.gaussian import Gaussian
from .membership_functions.iaa import IntervalAgreementApproach
from . import global_settings as gs


def _calculate_membership_values(data):
    """Calculate membership values for each data point.

    Each values membership is calculated as a proportion of how often
    it appears within the list of data points.
    """
    scale = Decimal(len(data))
    if gs.normalise_generated_sets:
        counter = Counter(data)
        scale = Decimal(counter.most_common()[0][1])
    else:
        scale = Decimal(len(data))
    return dict((gs.rnd(p), gs.rnd(data.count(p) / scale)) for p in set(data))


def generate_gaussian_t1_fuzzy_set(data):
    """Create a Gaussian distributed type-1 fuzzy set from the given data."""
    # mean and std won't work for Decimal data; ensure floats.
    float_data = [float(d) for d in data]
    return FuzzySet(Gaussian(Decimal(mean(float_data)),
                             Decimal(std(float_data))))





def generate_discrete_t1_fuzzy_set(data):
    """Create a discrete type-1 fuzzy set from the given data."""
    return DiscreteT1FuzzySet(_calculate_membership_values(data))


def generate_polling_t1_fuzzy_set(data):
    """Create a type-1 fuzzy set with interpolation from the given data."""
    return PollingT1FuzzySet(_calculate_membership_values(data))


def generate_polling_t2_fuzzy_set(data):
    """Create a type-1 fuzzy set with interpolation from the given data."""
    fs = T2AggregatedFuzzySet()
    for t1_subset in data:
        fs.add_membership_function(PollingT1FuzzySet(
                _calculate_membership_values(t1_subset)))
    return fs


def generate_iaa_t1_fuzzy_set(data):
    """Create a type-1 interval agreement approach set from interval data."""
    mf = IntervalAgreementApproach(gs.normalise_generated_sets)
    for d in data:
        mf.add_interval(d)
    return FuzzySet(mf)


def generate_iaa_t2_fuzzy_set(data):
    """Create a type-2 interval agreement approach set from interval data."""
    fs = T2AggregatedFuzzySet()
    for t1_subset in data:
        fs.add_membership_function(generate_iaa_t1_fuzzy_set(t1_subset))
    return fs

def generate_gaussian_t2_fuzzy_set(data):
    """Create a Gaussian distributed type-1 fuzzy set from the given data."""
    fs = T2AggregatedFuzzySet()
    for t1_subset in data:
        # mean and std won't work for Decimal data; ensure floats.
        float_subset = [float(d) for d in t1_subset]
        fs.add_membership_function(Gaussian(Decimal(mean(float_subset)),
                                            Decimal(std(float_subset))))
    return fs

def generate_gaussian_interval_t2_fuzzy_set(data):
    """Create a Gaussian distributed interval type-2 fuzzy set from the given data."""
    fs = T2AggregatedFuzzySet()
    for t1_subset in data:
        # mean and std won't work for Decimal data; ensure floats.
        float_subset = [float(d) for d in t1_subset]
        fs.add_membership_function(Gaussian(Decimal(mean(float_subset)),
                                            Decimal(std(float_subset))))
    return fs

