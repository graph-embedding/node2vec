import logging
import pandas as pd
from typing import Tuple
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import Row
from pyspark.sql import functions as ssf


def index_graph_pandas(df_graph: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Index all vertices and edges in a graph by using an int32 to represent a vertex

    :param df_graph: pandas dataframe of edge lists
    return the indexed DataFrame with vertex id and vertex name columns.
    """
    if "src" not in df_graph.columns or "dst" not in df_graph.columns:
        raise ValueError(f"Input graph NOT in the right format: {df_graph.columns}")
    if "weight" not in df_graph.columns:
        df_graph["weight"] = 1.0
    df_graph = df_graph[["src", "dst", "weight"]].astype({"weight": float})

    name_id = (
        df_graph[["src"]]
        .append(df_graph[["dst"]].rename(columns={"dst": "src"}), ignore_index=True,)
        .drop_duplicates()
        .reset_index()
    )
    name_id = name_id.rename(columns={"src": "vertex_name", "index": "vertex_id"})
    logging.info(f"Total number of vertices: {name_id.count()}")

    s_id = name_id.rename(columns={"vertex_name": "src", "vertex_id": "src_id"}).copy()
    d_id = name_id.rename(columns={"vertex_name": "dst", "vertex_id": "dst_id"}).copy()
    df_edge = df_graph.merge(s_id, on=["src"]).merge(d_id, on=["dst"])
    df_edge = df_edge[["src_id", "dst_id", "weight"]].rename(
        columns={"src_id": "src", "dst_id": "dst"}
    )
    logging.info(f"size of indexed graph: {df_edge.count()}")
    return name_id, df_edge


def index_graph_spark(
    spark: SparkSession, df_graph: DataFrame,
) -> Tuple[DataFrame, DataFrame]:
    """
    Index all vertices and edges in a graph by using an int32 to represent a vertex

    :param spark: an active SparkSession
    :param df_graph: Spark Dataframe of edge lists
    return the indexed DataFrame with vertex id and vertex name columns.
    """
    if "src" not in df_graph.columns or "dst" not in df_graph.columns:
        raise ValueError(f"Input graph NOT in the right format: {df_graph.columns}")
    if "weight" not in df_graph.columns:
        df_graph = df_graph.withColumn("weight", ssf.lit(1.0))
    df_graph = df_graph.withColumn("weight", df_graph["weight"].cast("float"))

    # name_id = (df_graph.select("src").union(df_graph.select("dst"))
    #            .distinct()
    #            .withColumnRenamed("src", "vertex_name")
    #            .withColumn("vertex_id", monotonically_increasing_id())
    #            ).cache()
    df = df_graph.select("src").union(df_graph.select("dst")).distinct().sort("src")
    name_id = (
        df.rdd.zipWithIndex()
        .map(lambda x: Row(vertex_id=x[1], vertex_name=x[0][0]))
        .toDF()
    ).cache()
    logging.info(f"Total num of vertices: {name_id.count()}")

    df_graph.createOrReplaceTempView("df")
    name_id.createOrReplaceTempView("name_id")
    df_edge = spark.sql(
        """
        SELECT a.vertex_id AS src,
               b.vertex_id AS dst,
               df.weight AS weight
          FROM df
          JOIN name_id AS a
            ON df.src = a.vertex_name
          JOIN name_id AS b
            ON df.dst = b.vertex_name
    """
    ).cache()
    logging.info(f"size of indexed graph: {df_edge.count()}")
    return name_id, df_edge
