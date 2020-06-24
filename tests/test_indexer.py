import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame


#
def test_index_graph_pandas():
    """
    test indexing func index_graph_pandas()
    """
    from node2vec.indexer import index_graph_pandas

    df_graph = pd.DataFrame.from_dict({
        'src': ['a1', 'a2', 'a3', 'a4'], 'dst': ['a2', 'b1', 'b2', 'a1'],
    })
    df, vid = index_graph_pandas(df_graph)

    assert isinstance(vid, pd.DataFrame) and len(vid) == 6
    assert isinstance(df, pd.DataFrame) and len(df) == len(df_graph)

    df1 = pd.DataFrame.from_dict({
        'dst': ['a2', 'b1', 'b2', 'a1'], 'weight': [0.8, 1.1, 1.0, 0.3]
    })
    pytest.raises(ValueError, index_graph_pandas, df1)


#
def test_index_graph_spark():
    """
    test indexing func index_graph_spark()
    """
    from node2vec.indexer import index_graph_spark

    spark = SparkSession.builder.config("spark.executor.cores", 4).getOrCreate()
    df_graph = spark.createDataFrame(pd.DataFrame.from_dict({
        'src': ['a1', 'a2', 'a3', 'a4'], 'dst': ['a2', 'b1', 'b2', 'a1'],
    }))
    df, vid = index_graph_spark(df_graph)

    assert isinstance(vid, DataFrame) and vid.count() == 6
    assert isinstance(df, DataFrame) and df.count() == df_graph.count()

    df1 = spark.createDataFrame(pd.DataFrame.from_dict({
        'dst': ['a2', 'b1', 'b2', 'a1'], 'weight': [0.8, 1.1, 1.0, 0.3]
    }))
    pytest.raises(ValueError, index_graph_spark, df1)
