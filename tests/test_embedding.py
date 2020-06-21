import os
import pytest
import shutil
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from gensim.models import KeyedVectors
from gensim.models import Word2Vec as GensimW2V
from pyspark.ml.feature import Word2VecModel as SparkW2VModel

from node2vec.embedding import Node2VecBase
from node2vec.embedding import Node2VecGensim
from node2vec.embedding import Node2VecSpark


#
def test_class_node2vecbase():
    """
    test class Node2VecBase
    """
    n2v = Node2VecBase()
    with pytest.raises(NotImplementedError):
        n2v.fit()
    with pytest.raises(NotImplementedError):
        n2v.get_vector()
    with pytest.raises(NotImplementedError):
        n2v.save_model("file:///a", "b")
    with pytest.raises(NotImplementedError):
        n2v.load_model("file:///a", "b")


def test_class_node2vecgensim():
    """
    test class Node2VecGensim
    """
    df = pd.DataFrame.from_dict({
        "walk": [[0, 1, 1, 0, 3, 4], [1, 2, 3, 2, 0, 4], [2, 3, 1, 0, 4, 4]]
    })
    n2v = Node2VecGensim(df)
    assert isinstance(n2v, Node2VecGensim)
    n2v = Node2VecGensim(
        df, w2v_params={"iter": 3}, window_size=6, vector_size=64, random_seed=1000,
    )
    assert isinstance(n2v, Node2VecGensim)
    pytest.raises(ValueError, Node2VecGensim, df, window_size=3)
    pytest.raises(ValueError, Node2VecGensim, df, vector_size=16)

    w2v_params = {"min_count": 0, "iter": 1, "seed": 1000, "batch_words": 1,
                  "size": 4, "workers": 4}
    n2v = Node2VecGensim(df, w2v_params=w2v_params)
    model = n2v.fit()
    assert isinstance(model, GensimW2V)

    assert isinstance(n2v.get_vector(), KeyedVectors)
    assert len(list(n2v.get_vector(vertex_id='0'))) > 0
    assert len(list(n2v.get_vector(vertex_id=1))) > 0

    n2v.save_model("./", "tmp")
    assert os.path.exists("./tmp.model")
    assert isinstance(n2v.load_model("./", "tmp"), GensimW2V)
    os.remove("./tmp.model")

    n2v.save_vectors("./", "tmp_vec")
    assert os.path.exists("./tmp_vec")
    assert isinstance(n2v.load_vectors("./", "tmp_vec"), KeyedVectors)
    os.remove("./tmp_vec")


def test_class_node2vecspark():
    """
    test class Node2VecSpark
    """
    spark = SparkSession.builder.config("spark.executor.cores", 4).getOrCreate()
    df = spark.createDataFrame(pd.DataFrame.from_dict({
        "walk": [[0, 1, 1, 0, 3, 4], [1, 2, 3, 2, 0, 4], [2, 3, 1, 0, 4, 4]]
    }))

    n2v = Node2VecSpark(df)
    assert isinstance(n2v, Node2VecSpark)
    n2v = Node2VecSpark(
        df, w2v_params={"maxIter": 3}, window_size=6, vector_size=64, random_seed=1000,
    )
    assert isinstance(n2v, Node2VecSpark)
    pytest.raises(ValueError, Node2VecSpark, df, window_size=3)
    pytest.raises(ValueError, Node2VecSpark, df, vector_size=16)

    w2v_params = {"minCount": 0, "maxIter": 1, "seed": 1000, "maxSentenceLength": 1,
                  "windowSize": 4}
    n2v = Node2VecSpark(df, w2v_params=w2v_params)
    model = n2v.fit()
    assert model is not None

    assert isinstance(n2v.get_vector(), DataFrame)
    assert len(list(n2v.get_vector(vertex_id=1))) > 0

    n2v.save_model("./", "tmp")
    assert os.path.exists("./tmp.sparkml")
    assert isinstance(n2v.load_model("./", "tmp"), SparkW2VModel)
    shutil.rmtree("./tmp.sparkml")
