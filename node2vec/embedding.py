import logging
import time
import gensim
import pandas as pd
import numpy as np
from typing import Union
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

    def get_vector(self, vertex_id: Optional[Union[str, int]] = None):
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

    f_model = "model.pckl"
    f_mdata = "metadata.json"

    def __init__(
        self,
        df_walks: pd.DataFrame,
        window_size: Optional[int] = None,
        vector_size: Optional[int] = None,
        w2v_params: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        A driver class for Node2Vec algorithm for vertex embedding, read and write
        vectors and/or models.

        :param df_walks: the 2-column dataframe of all random walk paths, [src, walk]
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
        self.model: Optional[GensimW2V] = None

        if w2v_params is not None:
            tmp_param = GENSIM_PARAMS.copy()
            tmp_param.update(w2v_params)
            w2v_params = tmp_param
        else:
            w2v_params = GENSIM_PARAMS
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

    def get_vector(self, vertex_id: Optional[Union[str, int]] = None) -> KeyedVectors:
        """
        Return vector associated with a node identified by the original node name/id
        """
        if vertex_id is None:
            return self.model.wv  # type: ignore
        if isinstance(vertex_id, int):
            vertex_id = str(vertex_id)
        return self.model.wv.__getitem__(vertex_id)  # type: ignore

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
        window_size: Optional[int] = None,
        vector_size: Optional[int] = None,
        w2v_params: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        A driver class for the distributed Node2Vec algorithm for vertex embedding,
        read and write vectors and/or models.

        :param df_walks: Spark dataframe of all random walk paths, [src, walk]
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
        self.model: Optional[SparkW2VModel] = None

        if w2v_params is not None:
            tmp_param = WORD2VEC_PARAMS.copy()
            tmp_param.update(w2v_params)
            w2v_params = tmp_param
        else:
            w2v_params = WORD2VEC_PARAMS
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

    def get_vector(self, vertex_id: Optional[Union[str, int]] = None) -> DataFrame:
        """
        :param vertex_id: the vertex name
        Return the vector associated with a node or all vectors
        """
        if vertex_id is None:
            return self.model.getVectors()  # type: ignore
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
