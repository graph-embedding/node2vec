import random
from pyspark.sql import SparkSession
from pyspark.sql import functions as ssf
from node2vec.spark import Node2VecSpark


if __name__ == "__main__":
    """
    This is a preset integration test on a small graph.
    Correctness test is not in the scope of this test.
    """
    spark = SparkSession.builder.appName("node2vec").getOrCreate()
    bucket = "gs://sq-dataproc-graphlib-prod"
    df = spark.read.csv(bucket + "/test_graph_10k/graph_edges.csv")
    df = df.toDF("src", "dst")
    df = df.withColumn("weight", ssf.lit(random.random()))
    g2v = Node2VecSpark(
        spark=spark,
        input_graph=df,
        indexed=True,
        directed=True,
        num_walks=2,
        walk_length=5,
        return_param=0.8,
        inout_param=1.5,
        vector_size=128,
        w2vparams=None,
    )
    df = g2v.embedding()
    df.write.parquet(bucket + "/test_graph_10k/result", "overwrite")
