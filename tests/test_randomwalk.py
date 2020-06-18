import random
import pytest
import numpy as np
import pandas as pd
from typing import List
from typing import Tuple

from node2vec.utils import Neighbors


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
    df = pd.DataFrame.from_dict({
        'src': src, 'dst': dst,
        'src_neighbors': src_neighbors,
        'dst_neighbors': dst_neighbors
    })
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


# #
# @pytest.mark.parametrize(
#     "path,code,dst_nbs,dst_wt,alias,probs,result",
#     [
#         ([-1, 0], u'gANdcQAoSv////9LAGUu', [1, 3], [1.0, 0.5],
#          [1, 0], [0.6666666666666666, 1.0], [0, 3]),
#         ([2, 1], u'gANdcQAoSwJLAWUu', [0, 2], [0.5, 1.0],
#          [0, 0], [1.0, 0.5714285714285715], [2, 1, 0]),
#         ([0, 3], u'gANdcQAoSwBLA2Uu', [0], [0.8], [0], [1.0], [0, 3, 0]),
#     ],
# )
# def test_initiate_random_walk(
#         path: List[int],
#         code: str,
#         dst_nbs: List[int],
#         dst_wt: List[float],
#         alias: List[int],
#         probs: List[float],
#         result: List[int],
# ):
#     """
#     test initiate_random_walk()
#     """
#     rp1 = RandomPath(path)
#     rp2 = RandomPath(code)
#     nb = Neighbors((dst_nbs, dst_wt))
#     ap = AliasProb((alias, probs))
#     for rp in [rp1, rp2]:
#         random.seed(20)
#         assert isinstance(rp, RandomPath)
#         assert rp.path == path
#         assert rp.last_edge == (path[-2], path[-1])
#         assert rp.serialize() == code
#         assert rp.__str__() == str(path)
#
#         assert rp.append(nb, ap).path == result
#
#
# #
# @pytest.mark.parametrize(
#     "nbs,result",
#     [
#         ([(1, 0.5), (1, 0.8), (3, 1.0)], ([2, 0, 1], [0.6521739, 1.0, 0.9565217])),
#         ([(0, 0.5), (2, 0.2)], ([0, 0], [1.0, 0.5714285714285715])),
#         ([(1, 0.2)], ([0], [1.0])),
#         ([(0, 1.0)], ([0], [1.0])),
#     ],
# )
# def test_next_step_random_walk(
#         nbs: List[Tuple[int, float]],
#         result: Tuple[List[int], List[float]],
# ) -> None:
#     """
#     test next_step_random_walk()
#     """
#     from node2vec.utils import generate_alias_tables
#
#     alias, probs = generate_alias_tables([w for _, w in nbs])
#     assert alias == result[0]
#     np.testing.assert_almost_equal(probs, result[1], decimal=7)
#
#
# #
# @pytest.mark.parametrize(
#     "nbs,result",
#     [
#         ([(1, 0.5), (1, 0.8), (3, 1.0)], ([2, 0, 1], [0.6521739, 1.0, 0.9565217])),
#         ([(0, 0.5), (2, 0.2)], ([0, 0], [1.0, 0.5714285714285715])),
#         ([(1, 0.2)], ([0], [1.0])),
#         ([(0, 1.0)], ([0], [1.0])),
#     ],
# )
# def test_to_path(
#         nbs: List[Tuple[int, float]],
#         result: Tuple[List[int], List[float]],
# ) -> None:
#     """
#     test to_path()
#     """
#     from node2vec.utils import generate_alias_tables
#
#     alias, probs = generate_alias_tables([w for _, w in nbs])
#     assert alias == result[0]
#     np.testing.assert_almost_equal(probs, result[1], decimal=7)
#
#
# @pytest.mark.parametrize(
#     "src_id,src_nbs,dst_nbs,param_p,param_q,result",
#     [
#         (0, [(1, 0.5), (2, 0.8), (3, 1.0)], [(0, 0.5), (2, 0.2)], 1.0, 1.0,
#          ([0, 0], [1.0, 0.5714285714285715])),
#         (1, [[(0, 0.5), (2, 0.2)]], [(1, 0.2)], 0.8, 1.5, ([0], [1.0])),
#         (3, [(0, 1.0)], [(1, 0.5), (3, 1.0)], 2.0, 4.0, ([1, 0], [0.4, 1.0])),
#     ],
# )
# def test_random_walk(
#         src_id: int,
#         src_nbs: List[Tuple[int, float]],
#         dst_nbs: List[Tuple[int, float]],
#         param_p: float,
#         param_q: float,
#         result: Tuple[List[int], List[float]],
# ) -> None:
#     """
#     test random_walk()
#     """
#     from node2vec.randomwalk import random_walk
#
#     alias, probs = generate_edge_alias_tables(
#         src_id, src_nbs, dst_nbs, param_p, param_q,
#     )
#     assert alias == result[0]
#     np.testing.assert_almost_equal(probs, result[1], decimal=7)
#
#     pytest.raises(ValueError, generate_edge_alias_tables, src_id, src_nbs, dst_nbs, 0)
#     pytest.raises(
#         ValueError, generate_edge_alias_tables, src_id, src_nbs, dst_nbs, 1.0, 0
#     )
