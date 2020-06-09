import pytest
import numpy as np


def test_map_edgelist_to_csrgraph():
    from graphlib.node2vec.numba import map_edgelist_to_csrgraph

    edge_list = [(0, 1, 0.5), (0, 3, 1.0), (1, 2, 0.2)]
    g, mp = map_edgelist_to_csrgraph(edge_list)
    assert mp == {0: 0, 1: 1, 3: 2, 2: 3}
    np.testing.assert_almost_equal(g.data, [0.5, 1.0, 0.2], decimal=5)
    np.testing.assert_equal(g.indptr, [0, 2, 3, 3, 3])
    np.testing.assert_equal(g.indices, [1, 2, 3])

    g, mp = map_edgelist_to_csrgraph(edge_list, mapnode=False)
    assert not mp
    np.testing.assert_almost_equal(g.data, [0.5, 1.0, 0.2], decimal=5)
    np.testing.assert_equal(g.indptr, [0, 2, 3, 3, 3])
    np.testing.assert_equal(g.indices, [1, 3, 2])

    edge_list = [(0, 1), (0, 4), (1, 3)]
    g, mp = map_edgelist_to_csrgraph(edge_list)
    assert mp == {0: 0, 1: 1, 4: 2, 3: 3}
    np.testing.assert_almost_equal(g.data, [1.0, 1.0, 1.0], decimal=5)
    np.testing.assert_equal(g.indptr, [0, 2, 3, 3, 3])
    np.testing.assert_equal(g.indices, [1, 2, 3])


def test_normalize_row():
    from graphlib.node2vec.numba import normalize_row

    weight = np.array([0.5, 1.0, 0.2])
    indptr = np.array([0, 2, 3, 3, 3])
    res = normalize_row(weight, indptr)
    np.testing.assert_almost_equal(res, [0.33333334, 0.6666667, 1.], decimal=5)

    weight = np.array([0.5, 1., 0.5, 0.2, 1., 0.2])
    indptr = np.array([0, 2, 4, 5, 6])
    res = normalize_row(weight, indptr)
    np.testing.assert_almost_equal(
        res, [0.33333334, 0.6666667, 0.71428573, 0.2857143, 1., 1.], decimal=5,
    )

    weight = np.array([0.5, 1.0, 0.2])
    indptr = np.array([0, 2, 3, 3, 4])
    with pytest.raises(ValueError):
        normalize_row(weight, indptr)


def test_unbiased_random_walk():
    from graphlib.node2vec.numba import unbiased_random_walk

    weight = np.array([0.33333334, 0.6666667, 0.71428573, 0.2857143, 1., 1.])
    indptr = np.array([0, 2, 4, 5, 6])
    colidx = np.array([1, 2, 0, 3, 0, 1])
    sampling_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    walk_length = 3
    paths = unbiased_random_walk(weight, indptr, colidx, sampling_nodes, walk_length)
    assert len(paths) == len(sampling_nodes)
    np.testing.assert_equal(
        [len(v) for v in paths], [walk_length] * len(sampling_nodes)
    )


def test_biased_random_walk():
    from graphlib.node2vec.numba import biased_random_walk

    weight = np.array([0.33333334, 0.6666667, 0.71428573, 0.2857143, 1., 1.])
    indptr = np.array([0, 2, 4, 5, 6])
    colidx = np.array([1, 2, 0, 3, 0, 1])
    sampling_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    walk_length = 3
    paths = biased_random_walk(
        weight, indptr, colidx, sampling_nodes, walk_length, 0.8, 1.5
    )
    assert len(paths) == len(sampling_nodes)
    np.testing.assert_equal(
        [len(v) for v in paths], [walk_length] * len(sampling_nodes)
    )
