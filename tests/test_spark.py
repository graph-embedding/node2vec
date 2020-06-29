import pytest
import numpy as np
import random
from typing import List
from typing import Tuple


@pytest.mark.parametrize(
    "neighbors,result",
    [
        ([0.5, 1.0], ([1, 0], [0.6666666666666666, 1.0])),
        ([0.5, 0.2], ([0, 0], [1.0, 0.5714286])),
        ([0.2], ([0], [1.0])),
        ([1.0], ([0], [1.0])),
    ],
)
def test_generate_alias_tables(
    neighbors: List[float],
    result: Tuple[List[int], List[float]],
) -> None:
    """
    test utils func generate_alias_tables()
    """
    from node2vec.spark import generate_alias_tables

    alias, probs = generate_alias_tables(neighbors)
    assert alias == result[0]
    np.testing.assert_almost_equal(probs, result[1], decimal=7)


@pytest.mark.parametrize(
    "src_id,shared_nb_ids,dst_nbs,return_param,inout_param,result",
    [
        (0, [2], ([0, 2], [0.5, 0.2]), 1.0, 1.0, ([0, 0], [1.0, 0.5714285714285715])),
        (1, [], ([1], [0.2]), 0.8, 1.5, ([0], [1.0])),
        (3, [], ([1, 3], [0.5, 1.0]), 2.0, 4.0, ([1, 0], [0.4, 1.0])),
    ],
)
def test_generate_edge_alias_tables(
    src_id: int,
    shared_nb_ids: List[int],
    dst_nbs: Tuple[List[int], List[float]],
    return_param: float,
    inout_param: float,
    result: Tuple[List[int], List[float]],
) -> None:
    """
    test utils func generate_edge_alias_tables()
    """
    from node2vec.spark import generate_edge_alias_tables

    alias, probs = generate_edge_alias_tables(
        src_id, shared_nb_ids, dst_nbs, return_param, inout_param,
    )
    assert alias == result[0]
    np.testing.assert_almost_equal(probs, result[1], decimal=7)


@pytest.mark.parametrize(
    "alias,probs,result",
    [
        ([1, 0], [0.6666666666666666, 1.0], 1),
        ([0, 0], [1.0, 0.5714285714285715], 0),
        ([0], [1.0], 0),
    ],
)
def test__sampling_from_alias(
        alias: List[int],
        probs: List[float],
        result: int,
) -> None:
    """
    test utils func _sampling_from_alias()
    """
    from node2vec.spark import _sampling_from_alias

    random.seed(20)
    ans = _sampling_from_alias(alias, probs, random.random(), random.random())
    assert ans == result


@pytest.mark.parametrize(
    "randv,alias,probs,result",
    [
        (0.14, [1, 0], [0.6666666666666666, 1.0], 0),
        (0.51, [0, 0], [1.0, 0.5714285714285715], 1),
        (0.88, [0], [1.0], 0),
    ],
)
def test__sampling_from_alias_wiki(
        randv: float,
        alias: List[int],
        probs: List[float],
        result: int,
) -> None:
    """
    test utils func _sampling_from_alias_wiki()
    """
    from node2vec.spark import _sampling_from_alias_wiki

    ans = _sampling_from_alias_wiki(alias, probs, randv)
    assert ans == result


@pytest.mark.parametrize(
    "path,dst_nbs,randv,alias,probs,result",
    [
        ([-1, 0], [1, 3], 0.14, [1, 0], [0.6666666666666666, 1.0], [0, 1]),
        ([2, 1], [0, 2], 0.51, [0, 0], [1.0, 0.5714285714285715], [2, 1, 2]),
        ([0, 3], [0], 0.88, [0], [1.0], [0, 3, 0]),
    ],
)
def test_extend_random_walk(
    path: List[int],
    dst_nbs: List[int],
    alias: List[int],
    probs: List[float],
    randv: float,
    result: List[int],
) -> None:
    """
    test utils func extend_random_walk()
    """
    from node2vec.spark import extend_random_walk

    ans = extend_random_walk(path, dst_nbs, alias, probs, randv)
    assert ans == result
