import random
import pytest
import numpy as np
import pandas as pd
from typing import List
from typing import Tuple

from node2vec.utils import Neighbors
from node2vec.utils import AliasProb
from node2vec.utils import RandomPath


#
def test_class_neighbors():
    """
    """
    random.seed(20)
    # tuple
    idx, weight = [0, 1, 2], [1.0, 0.2, 1.4]
    nb1 = Neighbors((idx, weight))
    df = pd.DataFrame.from_dict({'dst': idx, 'weight': weight})
    # str
    code64 = u'gANdcQAoSwBLAUsCZV1xAShHP/AAAAAAAABHP8mZmZmZmZpHP/ZmZmZmZmZlhnECLg=='
    nb2 = Neighbors(code64)
    #
    for nbs in [nb1, nb2]:
        assert isinstance(nbs, Neighbors)
        assert nbs.dst_id == idx
        assert nbs.dst_wt == weight

        assert list(nbs.items()) == [(0, 1.0), (1, 0.2), (2, 1.4)]
        assert nbs.serialize() == code64
        assert nbs.as_pandas().equals(df)

    # pandas df
    idx, weight = [1, 2, 3], [0.1, 1.2, 0.8]
    nbs = Neighbors(pd.DataFrame.from_dict({'dst': idx, 'weight': weight}))
    code64 = u'gANdcQAoSwFLAksDZV1xAShHP7mZmZmZmZpHP/MzMzMzMzNHP+mZmZmZmZplhnECLg=='
    df = pd.DataFrame.from_dict({'dst': idx, 'weight': weight})
    assert isinstance(nbs, Neighbors)
    assert nbs.dst_id == idx
    assert nbs.dst_wt == weight

    assert list(nbs.items()) == [(1, 0.1), (2, 1.2), (3, 0.8)]
    assert nbs.serialize() == code64
    assert nbs.as_pandas().equals(df)


#
def test_class_aliasprob():
    """
    """
    alias, probs = [1, 0], [0.6666666666666666, 1.0]
    code64 = u'gANdcQAoSwFLAGVdcQEoRz/lVVVVVVVVRz/wAAAAAAAAZYZxAi4='
    nbs = Neighbors(([11, 22], [1.0, 0.5]))
    jq1 = AliasProb((alias, probs))
    jq2 = AliasProb(pd.DataFrame.from_dict({'alias': alias, 'probs': probs}))
    jq3 = AliasProb(code64)
    #
    for jq in [jq1, jq2, jq3]:
        random.seed(20)
        assert isinstance(jq, AliasProb)
        assert jq.alias == alias
        assert jq.probs == probs
        assert jq.serialize() == code64

        assert jq.draw_alias(nbs) == 22

    #
    alias, probs = [0, 0], [1.0, 0.5714285714285715]
    code64 = u'gANdcQAoSwBLAGVdcQEoRz/wAAAAAAAARz/iSSSSSSSTZYZxAi4='
    nbs = Neighbors(([122, 221], [0.2, 1.5]))
    jq1 = AliasProb((alias, probs))
    jq2 = AliasProb(pd.DataFrame.from_dict({'alias': alias, 'probs': probs}))
    jq3 = AliasProb(code64)
    #
    for jq in [jq1, jq2, jq3]:
        random.seed(20)
        assert isinstance(jq, AliasProb)
        assert jq.alias == alias
        assert jq.probs == probs
        assert jq.serialize() == code64

        assert jq.draw_alias(nbs) == 122
        assert jq.draw_alias(nbs, 10) == 221


#
@pytest.mark.parametrize(
    "path,code,dst_nbs,dst_wt,alias,probs,result",
    [
        ([-1, 0], u'gANdcQAoSv////9LAGUu', [1, 3], [1.0, 0.5],
         [1, 0], [0.6666666666666666, 1.0], [0, 3]),
        ([2, 1], u'gANdcQAoSwJLAWUu', [0, 2], [0.5, 1.0],
         [0, 0], [1.0, 0.5714285714285715], [2, 1, 0]),
        ([0, 3], u'gANdcQAoSwBLA2Uu', [0], [0.8], [0], [1.0], [0, 3, 0]),
    ],
)
def test_class_randompath(
        path: List[int],
        code: str,
        dst_nbs: List[int],
        dst_wt: List[float],
        alias: List[int],
        probs: List[float],
        result: List[int],
) -> None:
    """
    """
    rp1 = RandomPath(path)
    rp2 = RandomPath(code)
    nb = Neighbors((dst_nbs, dst_wt))
    ap = AliasProb((alias, probs))
    for rp in [rp1, rp2]:
        random.seed(20)
        assert isinstance(rp, RandomPath)
        assert rp.path == path
        assert rp.last_edge == (path[-2], path[-1])
        assert rp.serialize() == code
        assert rp.__str__() == str(path)

        assert rp.append(nb, ap).path == result


#
@pytest.mark.parametrize(
    "nbs,result",
    [
        ([(1, 0.5), (1, 0.8), (3, 1.0)], ([2, 0, 1], [0.6521739, 1.0, 0.9565217])),
        ([(0, 0.5), (2, 0.2)], ([0, 0], [1.0, 0.5714285714285715])),
        ([(1, 0.2)], ([0], [1.0])),
        ([(0, 1.0)], ([0], [1.0])),
    ],
)
def test_generate_alias_tables(
        nbs: List[Tuple[int, float]],
        result: Tuple[List[int], List[float]],
) -> None:
    """
    """
    from node2vec.utils import generate_alias_tables

    alias, probs = generate_alias_tables([w for _, w in nbs])
    assert alias == result[0]
    np.testing.assert_almost_equal(probs, result[1], decimal=7)


@pytest.mark.parametrize(
    "src_id,src_nbs,dst_nbs,param_p,param_q,result",
    [
        (0, [(1, 0.5), (2, 0.8), (3, 1.0)], [(0, 0.5), (2, 0.2)], 1.0, 1.0,
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
    from node2vec.utils import generate_edge_alias_tables

    alias, probs = generate_edge_alias_tables(
        src_id, src_nbs, dst_nbs, param_p, param_q,
    )
    assert alias == result[0]
    np.testing.assert_almost_equal(probs, result[1], decimal=7)

    pytest.raises(ValueError, generate_edge_alias_tables, src_id, src_nbs, dst_nbs, 0)
    pytest.raises(
        ValueError, generate_edge_alias_tables, src_id, src_nbs, dst_nbs, 1.0, 0
    )
