
def estimate_pair_kuncheva_index(first_set, second_set, initial_num_of_features):
    """
    Compute the consistency index between two sets of features.
    Parameters
    ----------
    first_set: set
        First set of indices of selected features
    second_set: set
        Second set of indices of selected features
    initial_num_of_features: int
        Total number of features
    Returns
    -------
    cidx: float
        Consistency index between the two sets.
    Reference
    ---------
    Kuncheva, L.I. (2007). A Stability Index for Feature Selection.
    AIAC, pp. 390--395.
    """
    observed = float(len(first_set.intersection(second_set)))
    expected = len(first_set) * len(second_set) / float(initial_num_of_features)
    maxposbl = float(min(len(first_set), len(second_set)))
    cidx = -1.
    # It's 0 and not 1 as expected if num_features == len(sel1) == len(sel2) => observed = n
    # Because "take everything" and "take nothing" are trivial solutions we don't want to select
    if expected != maxposbl:
        cidx = (observed - expected) / (maxposbl - expected)

    return cidx


def estimate_total_kuncheva_index(feature_sets,  initial_num_of_features):
    """
    Compute the consistency index between more than 2 sets of features.
    This is done by averaging over all pairwise consistency indices.
    Parameters
    ----------
    feature_sets: list of lists
        List of k lists of indices of selected features
    initial_num_of_features: int
        Total number of features
    Returns
    -------
    cidx: float
        Consistency index between the k sets.
    Reference
    ---------
    Kuncheva, L.I. (2007). A Stability Index for Feature Selection.
    AIAC, pp. 390--395.
    """
    cidx = 0.
    for k1, sel1 in enumerate(feature_sets[:-1]):
        # sel_list[:-1] to not take into account the last list.
        # avoid a problem with sel_list[k1+1:] when k1 is the last element,
        # that give an empty list overwise
        # the work is done at the second to last element anyway
        for sel2 in feature_sets[k1 + 1:]:
            cidx += estimate_pair_kuncheva_index(set(sel1), set(sel2), initial_num_of_features)
    cidx = 2. * cidx / (len(feature_sets) * (len(feature_sets) - 1))

    return cidx
