import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as ssf
from fugue_spark import SparkExecutionEngine
from fugue_spark import SparkDataFrame
from node2vec.fugue import trim_index
from node2vec.fugue import random_walk
from node2vec.embedding import Node2VecSpark

logging.basicConfig(
    format='%(asctime)s %(levelname)s:: %(message)s', level=logging.INFO,
)


if __name__ == "__main__":
    """
    """
    spark = SparkSession.builder.appName("node2vec").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    fugue_spark = SparkExecutionEngine(spark)
    bucket = "[s3|gs]://[your input graph data]"  # loc of your input graph data
    # refer to constants.NODE2VEC_PARAMS for default values of n2v_params
    n2v_params = {
        "num_walks": 30, "walk_length": 10, "return_param": 1.0, "inout_param": 1.0,
    }
    # refer to constants.WORD2VEC_PARAMS for default values of w2v_params
    w2v_params = {}

    if len(sys.argv) <= 1 or sys.argv[1] == 'index':
        df = spark.read.parquet(f"{bucket}/input_graph.parquet").repartition(1000)
        df = df.select("src", "dst", "weight").withColumn("weight", ssf.log1p(df['weight']))
        fugue_df = SparkDataFrame(df.distinct())
        # assume the input graph is not indexed, and is directed
        df_index, name_id = trim_index(
            fugue_spark, fugue_df,  indexed=False, directed=True, max_out_deg=10000,
        )
        name_id.native.write.parquet(f"{bucket}/graph_name2id.parquet", "overwrite")
        df_index.native.write.parquet(f"{bucket}/graph_indexed.parquet", "overwrite")

    elif len(sys.argv) <= 1 or sys.argv[1] == 'walk':
        df = spark.read.parquet(f"{bucket}/graph_indexed.parquet").repartition(2000)
        fugue_df = SparkDataFrame(df.distinct())
        walks = random_walk(
            fugue_spark, fugue_df, n2v_params=n2v_params,
        )
        walks.native.write.parquet(f"{bucket}'graph_walks.parquet", "overwrite")

    else:
        df_walk = spark.read.parquet(f"{bucket}/graph_walks.parquet")
        name_id = spark.read.parquet(f"{bucket}/graph_name2id.parquet")
        g2v = Node2VecSpark(
            df_walk, w2v_params, name_id, window_size=5, vector_size=128,
        )
        model = g2v.fit()
        logging.info("model fitting done!")
        df = g2v.embedding()
        df.write.parquet(f"{bucket}'graph_embedding.parquet", "overwrite")
