import pytest
import numpy as np
from typing import List
from typing import Tuple


@pytest.mark.parametrize(
    "neighbors,result",
    [
        ([(), ()], ()),
        ([(), ()], ()),
        ([(), ()], ()),
    ],
)
def test_generate_alias_tables(
        neighbors: List[Tuple[int, float]],
        result: Tuple[List[int], List[float]],
) -> None:
    """

    """
    from node2vec.utils import generate_alias_tables

    alias, probs = generate_alias_tables(neighbors)
    assert alias == result[0]
    assert probs == result[1]


@pytest.mark.parametrize(
    "src_id,src_nbs,dst_nbs,param_p,param_q,result",
    [
        (1, [], [], 1.0, 1.0, ()),
        (2, [], [], 0.8, 1.5, ()),
        (3, [], [], 2.0, 4.0, ()),
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
    from node2vec.utils import generate_edge_alias_tables

    alias, probs = generate_edge_alias_tables(
        src_id, src_nbs, dst_nbs, param_p, param_q,
    )
    assert alias == result[0]
    assert probs == result[1]


@pytest.mark.parametrize(
    "alias,probs,result",
    [
        ([], [], 0),
        ([], [], 0),
        ([], [], 0),
    ],
)
def test_sampling_from_alias_original(
        alias: List[int],
        probs: List[float],
        result: int,
) -> None:
    """

    """
    from node2vec.utils import sampling_from_alias_original

    ans = sampling_from_alias_original(alias, probs)
    assert ans == result


@pytest.mark.parametrize(
    "seed,alias,probs,result",
    [
        (10.0, [], [], 0),
        (20.0, [], [], 0),
        (30.0, [], [], 0),
    ],
)
def test_sampling_from_alias(
        seed: float,
        alias: List[int],
        probs: List[float],
        result: int,
) -> None:
    """

    """
    from node2vec.utils import sampling_from_alias

    ans = sampling_from_alias(seed, alias, probs)
    assert ans == result


@pytest.mark.parametrize(
    "path,seed,dst_nbs,alias,probs,result",
    [
        ([], 100.0, [], [], [], []),
        ([], 200.0, [], [], [], []),
        ([], 300.0, [], [], [], []),
    ],
)
def test_next_step_random_walk(
        path: List[int],
        seed: float,
        dst_nbs: List[int],
        alias: List[int],
        probs: List[float],
        result: List[int],
) -> None:
    """

    """
    from node2vec.utils import next_step_random_walk

    ans = next_step_random_walk(
        path, seed, dst_nbs, alias, probs,
    )
    np.testing.assert_equal(ans, result)
