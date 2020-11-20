import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as ssf

from node2vec.spark import Node2VecSpark

logging.basicConfig(
    format='%(asctime)s %(levelname)s:: %(message)s', level=logging.INFO,
)


if __name__ == "__main__":
    """
    """
    spark = SparkSession.builder.appName("graph-embed").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    bucket = "[s3|gs]://[your input graph data]"  # loc of your input graph data
    # refer to constants.NODE2VEC_PARAMS for default values of n2v_params
    n2v_params = {
        "num_walks": 30, "walk_length": 10, "return_param": 1.0, "inout_param": 1.0,
    }
    # refer to constants.WORD2VEC_PARAMS for default values of w2v_params
    w2v_params = {}
    g2v = Node2VecSpark(
        spark, n2v_params, w2v_params=w2v_params, max_out_degree=10000,
    )

    if len(sys.argv) <= 1 or sys.argv[1] == "index":
        # the input graph must have 3 cols: src, dst, weight
        df = spark.read.parquet(f"{bucket}/input_graph.parquet").repartition(1000)
        df = df.select("src", "dst", "weight").withColumn("weight", ssf.log1p(df['weight']))
        # assume the input graph is not indexed, and is directed
        g2v.preprocess_input_graph(df, indexed=False, directed=True)
        g2v.name_id.write.parquet(f"{bucket}/graph_name2id.parquet", "overwrite")
        g2v.df.write.parquet(f"{bucket}/graph_indexed.parquet", "overwrite")

    elif sys.argv[1] == 'walk':
        g2v.name_id = spark.read.parquet(f"{bucket}/graph_name2id.parquet").cache()
        g2v.df = spark.read.parquet(f"{bucket}/graph_indexed.parquet").cache()
        walks = g2v.random_walk()
        walks.write.parquet(f"{bucket}/graph_walks.parquet", "overwrite")

    else:
        df_walks = spark.read.parquet(f"{bucket}/graph_walks.parquet")
        model = g2v.fit(df_walks)
        df_res = g2v.embedding()
        df_res.write.parquet(f"{bucket}/graph_embedding.parquet", "overwrite")
        logging.info("model fitting done!")
