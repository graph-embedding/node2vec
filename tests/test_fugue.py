from fugue import ArrayDataFrame
from fugue import NativeExecutionEngine
from fugue_spark import SparkDataFrame
from fugue_spark import SparkExecutionEngine
from pyspark.sql import SparkSession
from pyspark.sql import Row


#
def test_random_walk():
    """
    test random_walk()
    """
    from node2vec.fugue import random_walk

    graph = [[0, 2, 0.41], [0, 4, 0.85], [1, 5, 0.91], [2, 5, 0.3], [3, 4, 0.36],
             [3, 5, 0.3], [2, 0, 0.68], [4, 0, 0.1], [5, 1, 0.28], [5, 2, 0.88],
             [4, 3, 0.37], [5, 3, 0.97]]
    df = ArrayDataFrame(graph, schema="src:long,dst:long,weight:double")
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
