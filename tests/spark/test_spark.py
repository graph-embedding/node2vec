import random
import networkx as nx
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as ssf
from node2vec.spark.spark import Node2VecSpark


if __name__ == "__main__":
    """
    This is a preset integration test on a small graph.
    Correctness test is not in the scope of this test.
    """
    spark = SparkSession.builder.appName("node2vec-test").getOrCreate()
    sc = spark.sparkContext
    graph = nx.fast_gnp_random_graph(n=10, p=0.2)
    r = Row("src", "dst")
    data = [list(x) for x in graph.edges]
    df = sc.parallelize([r(*x) for x in data]).toDF()
    df = df.withColumn("weight", ssf.lit(random.random()))
    g2v = Node2VecSpark(
        spark=spark,
        input_graph=df,
        indexed=True,
        directed=False,
        num_walks=2,
        walk_length=3,
        return_param=0.8,
        inout_param=1.5,
        vector_size=128,
        w2vparams=None,
    )
    df = g2v.embedding()
    assert df.count() > 0
