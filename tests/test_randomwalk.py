import random
import json
import pytest
import numpy as np
import pandas as pd
from typing import List
from typing import Tuple

from node2vec.randomwalk import Neighbors
from node2vec.randomwalk import AliasProb
from node2vec.randomwalk import RandomPath


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
    "src_id,shd_ids,dst_nbs,param_p,param_q,result",
    [
        (0, [2], ([0, 2], [0.5, 0.2]), 1.0, 1.0, ([0, 0], [1.0, 0.5714285714285715])),
        (1, ([]), ([1], [0.2]), 0.8, 1.5, ([0], [1.0])),
        (3, ([]), ([1, 3], [0.5, 1.0]), 2.0, 4.0, ([1, 0], [0.4, 1.0])),
    ],
)
def test_generate_edge_alias_tables(
        src_id: int,
        shd_ids: List[int],
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
        src_id, shd_ids, dst_nbs, param_p, param_q,
    )
    assert alias == result[0]
    np.testing.assert_almost_equal(probs, result[1], decimal=7)

    # exception tests
    pytest.raises(ValueError, generate_edge_alias_tables, src_id, shd_ids, dst_nbs, 0)
    pytest.raises(
        ValueError, generate_edge_alias_tables, src_id, shd_ids, dst_nbs, 1.0, 0,
    )
    dst_nbs1 = (dst_nbs[0], dst_nbs[1][:-1])
    pytest.raises(ValueError, generate_edge_alias_tables, src_id, shd_ids, dst_nbs1)


#
#
def test_trim_hotspot_vertices():
    """
    test trim_hotspot_vertices()
    """
    from node2vec.randomwalk import trim_hotspot_vertices

    random.seed(20)
    src, dst, weight = [3, 3, 3], [0, 1, 2], [1.0, 0.2, 1.4]
    df = pd.DataFrame.from_dict({'src': src, 'dst': dst, 'weight': weight})
    res = trim_hotspot_vertices(df)
    src_, dst_, weight_ = [], [], []
    for row in res:
        src_.append(row["src"])
        dst_.append(row["dst"])
        weight_.append(row["weight"])
    assert src_ == src
    assert dst_ == dst
    assert weight_ == weight

    max_deg = 2
    res = trim_hotspot_vertices(df, max_out_degree=max_deg, random_seed=20)
    ground = set()
    for i in range(len(src)):
        ground.add(json.dumps({"src": src[i], "dst": dst[i], "weight": weight[i]}))
    n = 0
    for row in res:
        n += 1
        row["src"], row["dst"] = int(row["src"]), int(row["dst"])
        line = json.dumps(row)
        assert line in ground
    assert n == max_deg


#
def test_get_vertex_neighbors():
    """
    test calculate_vertex_attributes()
    """
    from node2vec.randomwalk import get_vertex_neighbors

    random.seed(20)
    src, dst, weight = [3, 3, 3], [0, 1, 2], [1.0, 0.2, 1.4]
    df = pd.DataFrame.from_dict({'src': src, 'dst': dst, 'weight': weight})
    code64 = u'gANdcQAoSwBLAUsCZV1xAShHP/AAAAAAAABHP8mZmZmZmZpHP/ZmZmZmZmZlhnECLg=='

    res = next(iter(get_vertex_neighbors(df)))
    assert sorted(res.keys()) == ['id', 'neighbors']
    assert res['id'] == src[0]
    assert res['neighbors'] == code64


#
def test_get_edge_shared_neighbors():
    """
    test calculate_edge_attributes()
    """
    from node2vec.randomwalk import get_edge_shared_neighbors

    random.seed(20)
    src, dst = [0, 0, 0], [1, 2, 3]
    dst_nbs = [Neighbors(([0, 2, 4], [0.5, 0.7, 1.0])),
               Neighbors(([0, 1, 3], [1.2, 0.7, 1.5])),
               Neighbors(([0, 2], [0.9, 1.5])),
               ]
    df = pd.DataFrame.from_dict(
        {'src': src, 'dst': dst, 'dst_neighbors': [nb.serialize() for nb in dst_nbs]}
    )

    num_walks = 2
    res = iter(get_edge_shared_neighbors(df, num_walks))
    for i in range(1, num_walks + 1):
        ans = next(res)
        assert sorted(ans.keys()) == ['dst', 'shared_neighbor_ids', 'src']
        assert ans['src'] == -i and ans['dst'] == 0
        assert ans['shared_neighbor_ids'] == []

    for i in range(len(df)):
        ans = next(res)
        shd_ids = [x for x in dst_nbs[i].dst_id if x in dst]
        assert sorted(ans.keys()) == ['dst', 'shared_neighbor_ids', 'src']
        assert ans['src'] == 0 and ans['dst'] == dst[i]
        assert ans['shared_neighbor_ids'] == shd_ids


def test_initiate_random_walk():
    """
    test initiate_random_walk()
    """
    from node2vec.randomwalk import initiate_random_walk

    random.seed(20)
    src, nbs, ap = [3, 2], [[0, 1, 2], [3, 1]], [u'abc', u'bcd']
    # df = pd.DataFrame.from_dict({'id': src, 'neighbors': nbs, 'alias_prob': ap})
    df = [{"dst": src[i], "neighbors": nbs[i]} for i in range(len(src))]

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
    shared_ids = [[2], [], []]
    df = [{"src": src[i], "dst": dst[i], "path": path[i],
           "dst_neighbors": dst_neighbors[i],
           "shared_neighbor_ids": shared_ids[i]} for i in range(len(src))]

    res = iter(next_step_random_walk(df, 1.0, 1.0))
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
