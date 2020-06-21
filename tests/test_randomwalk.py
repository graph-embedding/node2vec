import random
import pytest
import numpy as np
import pandas as pd
from typing import List
from typing import Tuple

from node2vec.randomwalk import Neighbors
from node2vec.randomwalk import AliasProb
from node2vec.randomwalk import RandomPath
from node2vec.randomwalk import generate_alias_tables


#
def test_class_neighbors():
    """
    test class Neighbors
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
    test class AliasProb
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
        assert jq.draw_alias(nbs, 10) == 22

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
    test class RandomPath
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
    test util func generate_alias_tables()
    """
    from node2vec.randomwalk import generate_alias_tables

    alias, probs = generate_alias_tables([w for _, w in nbs])
    assert alias == result[0]
    np.testing.assert_almost_equal(probs, result[1], decimal=7)


@pytest.mark.parametrize(
    "src_id,src_nbs,dst_nbs,param_p,param_q,result",
    [
        (0, ([1, 2, 3], [0.5, 0.8, 1.0]), ([0, 2], [0.5, 0.2]), 1.0, 1.0,
         ([0, 0], [1.0, 0.5714285714285715])),
        (1, ([0, 2], [0.5, 0.2]), ([1], [0.2]), 0.8, 1.5, ([0], [1.0])),
        (3, ([0], [1.0]), ([1, 3], [0.5, 1.0]), 2.0, 4.0, ([1, 0], [0.4, 1.0])),
    ],
)
def test_generate_edge_alias_tables(
        src_id: int,
        src_nbs: Tuple[List[int], List[float]],
        dst_nbs: Tuple[List[int], List[float]],
        param_p: float,
        param_q: float,
        result: Tuple[List[int], List[float]],
) -> None:
    """
    test util func generate_edge_alias_tables()
    """
    from node2vec.randomwalk import generate_edge_alias_tables

    # normal tests
    alias, probs = generate_edge_alias_tables(
        src_id, src_nbs, dst_nbs, param_p, param_q,
    )
    assert alias == result[0]
    np.testing.assert_almost_equal(probs, result[1], decimal=7)

    pytest.raises(ValueError, generate_edge_alias_tables, src_id, src_nbs, dst_nbs, 0)
    pytest.raises(
        ValueError, generate_edge_alias_tables, src_id, src_nbs, dst_nbs, 1.0, 0,
    )

    # exception tests
    src_nbs1 = (src_nbs[0][:-1], src_nbs[1])
    pytest.raises(ValueError, generate_edge_alias_tables, src_id, src_nbs1, dst_nbs)
    dst_nbs1 = (dst_nbs[0], dst_nbs[1][:-1])
    pytest.raises(ValueError, generate_edge_alias_tables, src_id, src_nbs, dst_nbs1)


#
#
#
def test_calculate_vertex_attributes():
    """
    test calculate_vertex_attributes()
    """
    from node2vec.randomwalk import calculate_vertex_attributes

    random.seed(20)
    src, dst, weight = [3, 3, 3], [0, 1, 2], [1.0, 0.2, 1.4]
    df = pd.DataFrame.from_dict({'src': src, 'dst': dst, 'weight': weight})
    code64 = u'gANdcQAoSwBLAUsCZV1xAShHP/AAAAAAAABHP8mZmZmZmZpHP/ZmZmZmZmZlhnECLg=='
    ap64 = u'gANdcQAoSwBLAksAZV1xAShHP/AAAAAAAABHP82J2J2J2J9HP+sTsTsTsTxlhnECLg=='

    res = next(iter(calculate_vertex_attributes(df)))
    assert sorted(res.keys()) == ['alias_prob', 'id', 'neighbors']
    assert res['id'] == src[0]
    assert res['neighbors'] == code64
    assert res['alias_prob'] == ap64


#
def test_calculate_edge_attributes():
    """
    test calculate_edge_attributes()
    """
    from node2vec.randomwalk import calculate_edge_attributes

    random.seed(20)
    src, dst = [0, 0], [1, 2]
    src_nbs = Neighbors(([1, 2, 3], [0.5, 1.2, 0.7]))
    src_neighbors = [src_nbs.serialize(), src_nbs.serialize()]
    dst_neighbors = [
        Neighbors(([0, 2, 4], [0.5, 0.9, 1.0])).serialize(),
        Neighbors(([0, 1], [1.2, 0.9])).serialize(),
    ]
    # df = pd.DataFrame.from_dict({
    #     'src': src, 'dst': dst,
    #     'src_neighbors': src_neighbors,
    #     'dst_neighbors': dst_neighbors
    # })
    df = [{'src': src[i], 'dst': dst[i], 'src_neighbors': src_neighbors[i],
           'dst_neighbors': dst_neighbors[i]} for i in range(len(src))]
    code64 = u'gANdcQAoSwFLAksDZV1xAShHP+AAAAAAAABHP/MzMzMzMzNHP+ZmZmZmZmZlhnECLg=='
    ap64 = u'gANdcQAoSwFLAEsBZV1xAShHP+QAAAAAAABHP/AAAAAAAABHP+wAAAAAAABlhnECLg=='

    num_walks = 2
    res = iter(calculate_edge_attributes(df, num_walks, 1.0, 1.0))
    for i in range(1, num_walks + 1):
        ans = next(res)
        assert sorted(ans.keys()) == ['alias_prob', 'dst', 'dst_neighbors', 'src']
        assert ans['src'] == -i and ans['dst'] == 0
        assert ans['dst_neighbors'] == code64
        assert ans['alias_prob'] == ap64

    ap64 = [u'gANdcQAoSwJLAEsBZV1xAShHP+QAAAAAAABHP/AAAAAAAABHP+wAAAAAAABlhnECLg==',
            u'gANdcQAoSwBLAGVdcQEoRz/wAAAAAAAARz/rbbbbbbbbZYZxAi4='
            ]
    for i in range(len(df)):
        ans = next(res)
        assert sorted(ans.keys()) == ['alias_prob', 'dst', 'dst_neighbors', 'src']
        assert ans['src'] == 0 and ans['dst'] == dst[i]
        assert ans['dst_neighbors'] == dst_neighbors[i]
        assert ans['alias_prob'] == ap64[i]


def test_initiate_random_walk():
    """
    test initiate_random_walk()
    """
    from node2vec.randomwalk import initiate_random_walk

    random.seed(20)
    src, nbs, ap = [3, 2], [[0, 1, 2], [3, 1]], [u'abc', u'bcd']
    # df = pd.DataFrame.from_dict({'id': src, 'neighbors': nbs, 'alias_prob': ap})
    df = [{"id": src[i], "neighbors": nbs[i], "alias_prob": ap[i]}
          for i in range(len(src))]

    num_walks = 3
    res = iter(initiate_random_walk(df, num_walks))
    for s_id in src:
        for i in range(num_walks):
            ans = next(res)
            assert sorted(ans.keys()) == ['dst', 'path', 'src']
            assert ans['src'] == -1 - i
            assert ans['dst'] == s_id
            assert ans['path'] == [-1 - i, s_id]


#
def test_next_step_random_walk():
    """
    test next_step_random_walk()
    """
    from node2vec.randomwalk import next_step_random_walk

    random.seed(20)
    src, dst = [0, 0, -1], [1, 2, 2]
    path = [[3, 0, 1], [2, 0, 2], [-1, 2]]
    dst_neighbors = [
        Neighbors(([0, 2, 4], [0.5, 0.9, 1.0])).serialize(),
        Neighbors(([0, 3], [1.2, 0.9])).serialize(),
        Neighbors(([0, 3], [1.2, 0.9])).serialize(),
    ]
    alias_prob = [
        AliasProb(generate_alias_tables([0.5, 0.9, 1.0])).serialize(),
        AliasProb(generate_alias_tables([1.2, 0.9])).serialize(),
        AliasProb(generate_alias_tables([1.2, 0.9])).serialize(),
    ]
    # df = pd.DataFrame.from_dict({
    #     'src': src,
    #     'dst': dst,
    #     'path': path,
    #     'dst_neighbors': dst_neighbors,
    #     'alias_prob': alias_prob,
    # })
    df = [{"src": src[i], "dst": dst[i], "path": path[i],
           "dst_neighbors": dst_neighbors[i], "alias_prob": alias_prob[i]}
          for i in range(len(src))]

    res = iter(next_step_random_walk(df))
    ans = next(res)
    assert sorted(ans.keys()) == ['dst', 'path', 'src']
    assert ans['src'] == 1
    assert ans['dst'] == 4
    assert ans['path'] == path[0] + [ans['dst']]

    random.seed(10)
    ans = next(res)
    assert sorted(ans.keys()) == ['dst', 'path', 'src']
    assert ans['src'] == 2
    assert ans['dst'] == 3
    assert ans['path'] == path[1] + [ans['dst']]

    random.seed(20)
    ans = next(res)
    assert sorted(ans.keys()) == ['dst', 'path', 'src']
    assert ans['src'] == 2
    assert ans['dst'] == 3
    assert ans['path'] == [2, 3]


#
def test_to_path():
    """
    test to_path()
    """
    from node2vec.randomwalk import to_path

    src, dst, path = [0, 1, 2], [2, 3, 4], [[1, 0, 2], [1, 3], [0, 2, 4]]
    # df = pd.DataFrame.from_dict({'src': src, 'dst': dst, 'path': path})
    df = [{'src': src[i], 'dst': dst[i], 'path': path[i]} for i in range(len(src))]
    res = iter(to_path(df))
    for i in range(len(src)):
        ans = next(res)
        assert sorted(ans.keys()) == ['src', 'walk']
        assert ans['src'] == path[i][0]
        assert ans['walk'] == path[i]
