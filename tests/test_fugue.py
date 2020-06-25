import pytest
import pandas as pd
from fugue import ArrayDataFrame
from fugue import PandasDataFrame
from fugue import NativeExecutionEngine
from fugue_spark import SparkDataFrame
from fugue_spark import SparkExecutionEngine
from pyspark.sql import SparkSession
from pyspark.sql import Row


#
def test_trim_index():
    """
    test Fugue func trim_index()
    """
    from node2vec.fugue import trim_index

    graph = [[0, 2, 0.41], [0, 4, 0.85], [3, 4, 0.36], [2, 0, 0.68], [4, 0, 0.1],
             [4, 3, 0.37]]
    df = ArrayDataFrame(graph, schema="src:int,dst:int,weight:double")

    df_res, name_id = trim_index(NativeExecutionEngine(), df, indexed=True)
    assert len(df_res.as_pandas()) == 6 and name_id is None
    df_res, name_id = trim_index(
        NativeExecutionEngine(), df, indexed=True, max_out_deg=1,
    )
    assert len(df_res.as_pandas()) == 4 and name_id is None

    spark = SparkSession.builder.config("spark.executor.cores", 4).getOrCreate()
    dat1 = {
        'src': ['a1', 'a1', 'a1', 'a2', 'b2'], 'dst': ['a2', 'b1', 'b2', 'b1', 'a2'],
    }
    dat2 = {
        'dst': ['a2', 'b1', 'b2', 'a1'], 'weight': [0.8, 1.1, 1.0, 0.3]
    }
    df = spark.createDataFrame(pd.DataFrame.from_dict(dat1))
    df_res, name_id = trim_index(
        SparkExecutionEngine(spark), SparkDataFrame(df), indexed=False, max_out_deg=2
    )
    assert df_res.count() == 4 and name_id.count() == 4
    df = spark.createDataFrame(pd.DataFrame.from_dict(dat2))
    pytest.raises(
        ValueError, trim_index, SparkExecutionEngine(spark), SparkDataFrame(df), True,
    )

    df = pd.DataFrame.from_dict(dat1)
    df_res, name_id = trim_index(
        NativeExecutionEngine(), PandasDataFrame(df), indexed=False,
    )
    assert len(df_res.as_pandas()) == 5 and len(name_id.as_pandas()) == 4
    df = pd.DataFrame.from_dict(dat2)
    pytest.raises(
        ValueError, trim_index, NativeExecutionEngine(), PandasDataFrame(df), False,
    )


#
def test_random_walk():
    """
    test Fugue func random_walk()
    """
    from node2vec.fugue import random_walk

    graph = [[0, 2, 0.41], [0, 4, 0.85], [3, 4, 0.36], [2, 0, 0.68], [4, 0, 0.1],
             [4, 3, 0.37]]
    df = ArrayDataFrame(graph, schema="src:int,dst:int,weight:double")
    n2v_params = {"num_walks": 2, "walk_length": 3, "return_param": 0.5}

    res = random_walk(NativeExecutionEngine(), df, n2v_params)
    assert res is not None
    res = random_walk(NativeExecutionEngine(), df.as_pandas(), n2v_params)
    assert res is not None

    spark = SparkSession.builder.config("spark.executor.cores", 4).getOrCreate()
    r = Row("src", "dst", "weight")
    df = spark.sparkContext.parallelize([r(*x) for x in graph]).toDF()
    res = random_walk(SparkExecutionEngine(spark), SparkDataFrame(df), n2v_params)
    assert res is not None
