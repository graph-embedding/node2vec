import logging
import time
import gensim
import pandas as pd
import numpy as np
from typing import Union
from typing import List
from typing import Any
from typing import Dict
from typing import Optional
from pyspark.sql import DataFrame
from gensim.models import KeyedVectors
from gensim.models import Word2Vec as GensimW2V
from pyspark.ml.feature import Word2Vec as SparkW2V
from pyspark.ml.feature import Word2VecModel as SparkW2VModel

from node2vec.constants import WORD2VEC_PARAMS
from node2vec.constants import GENSIM_PARAMS


#
class Node2VecBase(object):
    """
    Base class for conducting Node2Vec in various computing frameworks
    """

    def __init__(self):
        pass

    def fit(self):
        """
        the entry point for fitting a node2vec process for vertex embedding
        Return the fit model of Word2Vec
        """
        raise NotImplementedError()

    def embedding(self):
        """
        Return the final embedding results as a 2-col dataframe. If indexing is
        conducted, this func will map vertex id back to the original vertex name.
        """
        raise NotImplementedError()

    def get_vector(self, vertex_id: Union[str, int]):
        """
        vertex_id: str or int, either the node ID or name depending on graph format
        Return vector associated with a node identified by the original node name/id
        """
        raise NotImplementedError()

    def save_model(self, file_path: str, file_name: str):
        """
        Saves the word2vec model object to a cloud bucket, always overwrite.
        :param file_path: a gcs or s3 bucket, or local folder
        :param file_name: the name to be used for the model file
        """
        raise NotImplementedError()

    def load_model(self, file_path: str, file_name: str):
        """
        Load a previously saved Word2Vec model object to memory.
        :param file_path: a gcs or s3 bucket, or local folder
        :param file_name: the name to be used for the model file, append ".model" to
                           it if it doesn't end with ".model".
        """
        raise NotImplementedError()


#
class Node2VecGensim(Node2VecBase):
    """
    A wrapper class to handle input and output data for distributed node2vec algorithm.
    """

    def __init__(
        self,
        df_walks: pd.DataFrame,
        w2v_params: Dict[str, Any],
        name_id: Optional[pd.DataFrame] = None,
        window_size: Optional[int] = None,
        vector_size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        A driver class for Node2Vec algorithm for vertex embedding, read and write
        vectors and/or models.

        :param df_walks: the 2-column dataframe of all random walk paths, [src, walk]
        :param name_id: a two-col dataframe of mapping from vertex name to vertex id
        :param window_size: the context size for word2vec embedding
        :param vector_size: int, dimension of the output graph embedding representation
                            num of codes after transforming from words (dimension
                            of embedding feature representation), usually power of 2,
                            e.g. 64, 128, 256
        :param w2v_params: dict of parameters to pass to gensim's word2vec module (not
                           to set embedding dim here)
        :param random_seed: optional random seed, for testing only
        """
        logging.info("__init__(): preprocssing in spark ...")
        super().__init__()
        self.walks = df_walks
        self.name_id = name_id
        self.model: Optional[GensimW2V] = None

        for param in GENSIM_PARAMS:
            if param not in w2v_params:
                w2v_params[param] = GENSIM_PARAMS[param]
        w2v_params["seed"] = random_seed if random_seed else int(time.time()) // 60
        if window_size is not None:
            if window_size < 5 or window_size > 40:
                raise ValueError(f"Inappropriate context window size {window_size}!")
            w2v_params["window"] = window_size
        if vector_size is not None:
            if vector_size < 32 or vector_size > 1024:
                raise ValueError(f"Inappropriate vector dimension {vector_size}!")
            w2v_params["size"] = vector_size
        logging.info(f"__init__(): w2v params: {w2v_params}")
        self.w2v_params = w2v_params

    def fit(self) -> GensimW2V:
        """
        the entry point for fitting a node2vec process for graph feature embedding
        Returns a gensim model of Word2Vec
        """
        all_walks = np.array(self.walks["walk"].tolist()).astype(str).tolist()
        self.model = gensim.models.Word2Vec(sentences=all_walks, **self.w2v_params)
        return self.model

    def embedding(self) -> pd.DataFrame:
        """
        Return the resulting df, and map back vertex name if the graph is indexed
        """
        if self.model is None:
            raise ValueError("Model is not available. Please run fit()")
        ids = [int(t) for t in self.model.wv.vocab]
        vectors = [list(self.model.wv.__getitem__(t)) for t in self.model.wv.vocab]
        if self.name_id is not None:
            dic = self.name_id.set_index("id").to_dict()["name"]
            names = [dic[i] for i in ids]
            df_res = pd.DataFrame.from_dict({"name": names, "vector": vectors})
        else:
            df_res = pd.DataFrame.from_dict({"id": ids, "vector": vectors})
        return df_res

    def get_vector(self, vertex_id: Union[str, int]) -> List[float]:
        """
        Return vector associated with a node identified by the original node name/id
        """
        if isinstance(vertex_id, int):
            vertex_id = str(vertex_id)
        return list(self.model.wv.__getitem__(vertex_id))  # type: ignore

    def save_model(self, file_path: str, file_name: str) -> None:
        """
        Save model of gensim.models.Word2Vec into a folder
        """
        self.model.save(file_path + "/" + file_name + ".model")  # type: ignore

    def load_model(self, file_path: str, file_name: str) -> GensimW2V:
        """
        Load trained model of gensim.models.Word2Vec to the model
        """
        self.model = GensimW2V.load(file_path + "/" + file_name + ".model")
        return self.model

    def save_vectors(self, file_path: str, file_name: str) -> None:
        """
        Save as embeddings in gensim.models.KeyedVectors format
        """
        self.model.wv.save_word2vec_format(file_path + "/" + file_name)  # type: ignore

    @staticmethod
    def load_vectors(file_path: str, file_name: str) -> KeyedVectors:
        """
        Load embeddings from gensim.models.KeyedVectors format
        """
        model_wv = KeyedVectors.load_word2vec_format(file_path + "/" + file_name)
        return model_wv


#
class Node2VecSpark(Node2VecBase):
    """
    A wrapper class to handle input and output data for distributed node2vec algorithm.
    """

    def __init__(
        self,
        df_walks: DataFrame,
        w2v_params: Dict[str, Any],
        name_id: Optional[DataFrame] = None,
        window_size: Optional[int] = None,
        vector_size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        A driver class for the distributed Node2Vec algorithm for vertex embedding,
        read and write vectors and/or models.

        :param df_walks: Spark dataframe of all random walk paths, [src, walk]
        :param w2v_params: dict of parameters to pass to gensim's word2vec module (not
                           to set embedding dim here)
        :param name_id: a two-col dataframe of mapping from vertex name to vertex id
        :param window_size: the context size for word2vec embedding
        :param vector_size: int, dimension of the output graph embedding representation
                            num of codes after transforming from words (dimension
                            of embedding feature representation), usually power of 2,
                            e.g. 64, 128, 256
        :param random_seed: optional random seed, for testing only
        """
        logging.info("__init__(): preprocssing in spark ...")
        super().__init__()
        self.walks = df_walks
        self.name_id = name_id
        self.model: Optional[SparkW2VModel] = None

        # update w2v_params
        for param in WORD2VEC_PARAMS:
            if param not in w2v_params:
                w2v_params[param] = WORD2VEC_PARAMS[param]
        w2v_params["seed"] = random_seed if random_seed else int(time.time())
        if window_size is not None:
            if window_size < 5 or window_size > 40:
                raise ValueError(f"Inappropriate context window size {window_size}!")
            w2v_params["windowSize"] = window_size
        if vector_size is not None:
            if vector_size < 32 or vector_size > 1024:
                raise ValueError(f"Inappropriate vector dimension {vector_size}!")
            w2v_params["vectorSize"] = vector_size
        logging.info(f"__init__(): w2v params: {w2v_params}")
        self.word2vec = SparkW2V(inputCol="walk", outputCol="vector", **w2v_params)

    def fit(self) -> SparkW2VModel:
        """
        fit the word2vec model in spark
        """
        df_walks = self.walks.select("walk").withColumn(
            "walk", self.walks["walk"].cast("array<string>")
        )
        self.model = self.word2vec.fit(df_walks)
        logging.info("model fitting done!")
        return self.model

    def embedding(self) -> DataFrame:
        """
        Return the resulting df, and map back vertex name if the graph is indexed
        """
        if self.model is None:
            raise ValueError("Model is not available. Please run fit()")
        df = self.model.getVectors().withColumnRenamed("word", "id")
        if self.name_id is not None:
            df = df.join(self.name_id, on=["id"]).select("name", "vector")
        return df

    def get_vector(self, vertex_id: Union[str, int]) -> DataFrame:
        """
        :param vertex_id: the vertex name
        Return the vector associated with a node or all vectors
        """
        return self.model.getVectors().filter(f"word = {vertex_id}")  # type: ignore

    def save_model(self, cloud_path: str, model_name: str,) -> None:
        """
        Saves the word2vec model object to a cloud bucket, always overwrite.
        """
        if not model_name.endswith(".sparkml"):
            model_name += ".sparkml"
        self.model.save(cloud_path + "/" + model_name)  # type: ignore

    def load_model(self, cloud_path: str, model_name: str,) -> SparkW2VModel:
        """
        Load a previously saved Word2Vec model object to memory.
        """
        if not model_name.endswith(".sparkml"):
            model_name += ".sparkml"
        self.model = SparkW2VModel.load(cloud_path + "/" + model_name)
        return self.model
