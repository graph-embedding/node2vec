import random
import pandas as pd
from fugue import ArrayDataFrame
from fugue_spark import SparkDataFrame
from fugue import NativeExecutionEngine
from fugue_spark import SparkExecutionEngine
from pyspark.sql import SparkSession
from node2vec.utils import Neighbors
from node2vec.utils import AliasProb
from node2vec.utils import generate_alias_tables


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


def test_initiate_random_walk():
    """
    test initiate_random_walk()
    """
    from node2vec.randomwalk import initiate_random_walk

    random.seed(20)
    src, nbs, ap = [3, 2], [[0, 1, 2], [3, 1]], [u'abc', u'bcd']
    df = pd.DataFrame.from_dict({'id': src, 'neighbors': nbs, 'alias_prob': ap})

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
    df = pd.DataFrame.from_dict({
        'src': src,
        'dst': dst,
        'path': path,
        'dst_neighbors': dst_neighbors,
        'alias_prob': alias_prob,
    })

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
    df = pd.DataFrame.from_dict({'src': src, 'dst': dst, 'path': path})
    res = iter(to_path(df))
    for i in range(len(src)):
        ans = next(res)
        assert sorted(ans.keys()) == ['src', 'walk']
        assert ans['src'] == path[i][0]
        assert ans['walk'] == path[i]


#
def test_random_walk():
    """
    test random_walk()
    """
    from node2vec.randomwalk import random_walk

    graph = [[0, 2, 0.41], [0, 4, 0.85], [1, 5, 0.91], [2, 5, 0.3], [3, 4, 0.36],
             [3, 5, 0.3], [2, 0, 0.68], [4, 0, 0.1], [5, 1, 0.28], [5, 2, 0.88],
             [4, 3, 0.37], [5, 3, 0.97]]
    df = ArrayDataFrame(graph, schema="src:long,dst:long,weight:double")
    n2v_params = {"num_walks": 2, "walk_length": 3, "return_param": 0.5}

    res = random_walk(NativeExecutionEngine(), df, n2v_params)
    assert res is not None

    spark = SparkSession.builder.config("spark.executor.cores", 4).getOrCreate()
    spark_df = spark.createDataFrame(df.as_pandas())
    res = random_walk(SparkExecutionEngine(spark), spark_df, n2v_params)
    assert res is not None
