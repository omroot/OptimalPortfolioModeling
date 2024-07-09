import numpy as np
import pandas as pd


def quasi_diagonalize_linkage(link: np.ndarray)-> list:
    """sort clustered items by distance

    """
    link = link.astype(int)
    # get the first and the second item of the last tuple
    ordered_items_indicies = pd.Series([link[-1,0], link[-1,1]])
    # the total number of original items is the third item of the last list
    number_items = link[-1, 3]
    # if the max of ordered_items_indicies is bigger than or equal to the number of original items
    while ordered_items_indicies.max() >= number_items:
        # make space
        ordered_items_indicies.index = range(0, ordered_items_indicies.shape[0] * 2, 2)
        # locate clusters
        clusters = ordered_items_indicies[ordered_items_indicies >= number_items]
        i = clusters.index
        j = clusters.values - number_items
        # item 1
        ordered_items_indicies[i] = link[j, 0]
        # item 2
        clusters = pd.Series(link[j, 1], index=i + 1)
        ordered_items_indicies = ordered_items_indicies.append(clusters)
        # re-sort
        ordered_items_indicies = ordered_items_indicies.sort_index()
        # re-index
        ordered_items_indicies.index = range(ordered_items_indicies.shape[0])

    return ordered_items_indicies.tolist()


