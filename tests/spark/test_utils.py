import pytest
import numpy as np
import random
from typing import List
from typing import Tuple

random.seed(20)


@pytest.mark.parametrize(
    "neighbors,result",
    [
        ([(1, 0.5), (3, 1.0)], ([1, 0], [0.6666666666666666, 1.0])),
        ([(0, 0.5), (2, 0.2)], ([0, 0], [1.0, 0.5714285714285715])),
        ([(1, 0.2)], ([0], [1.0])),
        ([(0, 1.0)], ([0], [1.0])),
    ],
)
def test_generate_alias_tables(
        neighbors: List[Tuple[int, float]],
        result: Tuple[List[int], List[float]],
) -> None:
    """

    """
    from node2vec.spark.utils import generate_alias_tables

    alias, probs = generate_alias_tables(neighbors)
    assert alias == result[0]
    np.testing.assert_almost_equal(probs, result[1], decimal=7)


@pytest.mark.parametrize(
    "neighbors,result",
    [
        ([(1, 0.5), (3, 1.0)], ([1, 1], [0.6666666666666666, 1.0])),
        ([(0, 0.5), (2, 0.2)], ([0, 0], [1.0, 0.5714285714285715])),
        ([(1, 0.2)], ([0], [1.0])),
        ([(0, 1.0)], ([0], [1.0])),
    ],
)
def test_generate_alias_tables_wiki(
        neighbors: List[Tuple[int, float]],
        result: Tuple[List[int], List[float]],
) -> None:
    """

    """
    from node2vec.spark.utils import generate_alias_tables_wiki

    alias, probs = generate_alias_tables_wiki(neighbors)
    assert alias == result[0]
    np.testing.assert_almost_equal(probs, result[1], decimal=7)


@pytest.mark.parametrize(
    "src_id,src_nbs,dst_nbs,param_p,param_q,result",
    [
        (0, [(1, 0.5), (3, 1.0)], [(0, 0.5), (2, 0.2)], 1.0, 1.0,
         ([0, 0], [1.0, 0.5714285714285715])),
        (1, [[(0, 0.5), (2, 0.2)]], [(1, 0.2)], 0.8, 1.5, ([0], [1.0])),
        (3, [(0, 1.0)], [(1, 0.5), (3, 1.0)], 2.0, 4.0, ([1, 0], [0.4, 1.0])),
    ],
)
def test_generate_edge_alias_tables(
        src_id: int,
        src_nbs: List[Tuple[int, float]],
        dst_nbs: List[Tuple[int, float]],
        param_p: float,
        param_q: float,
        result: Tuple[List[int], List[float]],
) -> None:
    """

    """
    from node2vec.spark.utils import generate_edge_alias_tables

    alias, probs = generate_edge_alias_tables(
        src_id, src_nbs, dst_nbs, param_p, param_q,
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
def test_sampling_from_alias_original(
        alias: List[int],
        probs: List[float],
        result: int,
) -> None:
    """

    """
    from node2vec.spark.utils import sampling_from_alias_original

    ans = sampling_from_alias_original(alias, probs)
    assert ans == result


@pytest.mark.parametrize(
    "randv,alias,probs,result",
    [
        (0.14, [1, 0], [0.6666666666666666, 1.0], 0),
        (0.51, [0, 0], [1.0, 0.5714285714285715], 1),
        (0.88, [0], [1.0], 0),
    ],
)
def test_sampling_from_alias(
        randv: float,
        alias: List[int],
        probs: List[float],
        result: int,
) -> None:
    """

    """
    from node2vec.spark.utils import sampling_from_alias

    ans = sampling_from_alias(randv, alias, probs)
    assert ans == result


@pytest.mark.parametrize(
    "path,dst_nbs,randv,alias,probs,result",
    [
        ([-1, 0], [1, 3], 0.14, [1, 0], [0.6666666666666666, 1.0], [0, 1]),
        ([2, 1], [0, 2], 0.51, [0, 0], [1.0, 0.5714285714285715], [2, 1, 2]),
        ([0, 3], [0], 0.88, [0], [1.0], [0, 3, 0]),
    ],
)
def test_next_step_random_walk(
        path: List[int],
        dst_nbs: List[int],
        randv: float,
        alias: List[int],
        probs: List[float],
        result: List[int],
) -> None:
    """

    """
    from node2vec.spark.utils import next_step_random_walk

    ans = next_step_random_walk(path, randv, dst_nbs, alias, probs)
    assert ans == result
