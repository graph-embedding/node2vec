import logging
import os
import sys
import time
import json
import joblib
import shutil
import sklearn
import tempfile
import gensim
import pandas as pd
from typing import Union
from typing import Any
from typing import Dict
from typing import Optional
from pyspark.sql import DataFrame
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

    def save_vectors(self, file_path: str, file_name: str):
        """
        Save as graph embedding vectors as a dataFrame
        :param file_path: a gcs or s3 bucket, or local folder
        :param file_name: the name to be used for the model file
        """
        raise NotImplementedError()

    def load_vectors(self, file_path: str, file_name: str):
        """
        Load graph embedding vectors from saved file to a dataFrame
        :param file_path: a gcs or s3 bucket, or local folder
        :param file_name: the name to be used for the model file
        returns a dataframe
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

        if w2v_params is None:
            w2v_params = GENSIM_PARAMS
        if w2v_params.get("seed", None) is None:
            w2v_params["seed"] = random_seed or int(round(time.time() * 1000))
        if window_size is not None:
            if window_size < 5 or window_size > 40:
                raise ValueError(f"Inappropriate context window size {window_size}!")
            w2v_params["windowSize"] = window_size
        if vector_size is not None:
            if vector_size < 32 or vector_size > 1024:
                raise ValueError(f"Inappropriate vector dimension {vector_size}!")
            w2v_params["vectorSize"] = vector_size
        logging.info(f"__init__(): w2v params: {w2v_params}")
        self.w2v_params = w2v_params

    def fit(self) -> GensimW2V:
        """
        the entry point for fitting a node2vec process for graph feature embedding
        Returns a gensim model of Word2Vec
        """
        all_walks = self.walks.astype(str).tolist()
        self.model = gensim.models.Word2Vec(sentences=all_walks, **self.w2v_params)
        return self.model

    def get_vector(self, vertex_id: Optional[Union[str, int]] = None) -> Any:
        """
        Return vector associated with a node identified by the original node name/id
        """
        if vertex_id is None:
            return self.model.wv  # type: ignore
        return self.model.wv.__getitem__(vertex_id)  # type: ignore

    def save(self, file_path: str, file_name: str) -> None:
        """
        Save as embeddings in gensim.models.KeyedVectors format
        """
        self.model.save(file_path + "/" + file_name + ".model")  # type: ignore

    def load(self, file_path: str, file_name: str) -> GensimW2V:
        """
        Load embeddings from gensim.models.KeyedVectors format from
        """
        self.model = GensimW2V.load(file_path + "/" + file_name + ".model")
        return self.model

    def save_vectors(self, file_path: str, file_name: str) -> None:
        """
        Save as embeddings in gensim.models.KeyedVectors format
        """
        self.model.wv.save_word2vec_format(file_path + "/" + file_name)  # type: ignore

    def load_vectors(self, file_path: str, file_name: str) -> None:
        """
        Load embeddings from gensim.models.KeyedVectors format from
        """
        self.model = gensim.wv.load_word2vec_format(file_path + "/" + file_name)

    def save_model(self, file_path: str, file_name: str) -> None:
        """
        Saves the word2vec model to a zipfile with joblib dump (pickle-like) +
        dependency metadata. Metadata is checked on load. Includes validation and
        metadata to avoid Pickle deserialization gotchas.

        Refer to Alex Gaynor PyCon 2014 talk "Pickles are for Delis" for the reason of
        this additional check
        """
        sysverinfo = sys.version_info
        meta_data = {
            "python_": f"{sysverinfo[0]}.{sysverinfo[1]}",
            "skl_": sklearn.__version__[:-2],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            joblib.dump(self, os.path.join(temp_dir, self.f_model), compress=True)
            with open(os.path.join(temp_dir, self.f_mdata), "w") as f:
                json.dump(meta_data, f)
            shutil.make_archive(file_path + "/" + file_name, "zip", temp_dir)

    def load_model(self, file_path: str, file_name: str) -> Any:
        """
        Load model from NodeEmbedding model zip file.
        Loading checks for metadata and raises ValueError if pkg versions don't match.

        Returns the model object.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(file_path + ".zip", temp_dir, "zip")
            self.model = joblib.load(os.path.join(temp_dir, self.f_model))

            # Validate the metadata
            with open(os.path.join(temp_dir, self.f_mdata)) as f:
                meta_data = json.load(f)
            pyver = "{0}.{1}".format(sys.version_info[0], sys.version_info[1])
            if meta_data["python_"] != pyver:
                raise ValueError(
                    f"Python version {pyver} NOT match metadata {meta_data['python_']}!"
                )
            sklver = sklearn.__version__[:-2]
            if meta_data["skl_"] != sklver:
                raise ValueError(
                    f"sklearn version {sklver} NOT match metadata {meta_data['skl_']}"
                )
            return self.model


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
        self.model: Optional[SparkW2VModel] = None

        if w2v_params is None:
            w2v_params = WORD2VEC_PARAMS
        if w2v_params.get("seed", None) is None:
            w2v_params["seed"] = random_seed or int(round(time.time() * 1000))
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

    def get_vector(self, vertex_name: Optional[Union[str, int]] = None) -> DataFrame:
        """
        :param vertex_name: the vertex name
        Return the vector associated with a node or all vectors
        """
        if self.model is None:
            self.fit()
        if vertex_name is None:
            return self.model.getVectors()  # type: ignore
        return self.model.getVectors().filter(f"word = {vertex_name}")  # type: ignore

    def save_vectors(self, cloud_path: str, model_name: str,) -> None:
        """
        Save as graph embedding vectors as a Spark DataFrame
        """
        try:
            if not cloud_path.endswith("/"):
                cloud_path += "/"
            self.word2vec.save(cloud_path + model_name)
        except Exception as e:
            raise ValueError("save_vectors(): failed with exception!") from e

    def load_vectors(self, cloud_path: str, model_name: str,) -> DataFrame:
        """
        Load graph embedding vectors from saved file to a Spark DataFrame
        """
        try:
            if not cloud_path.endswith("/"):
                cloud_path += "/"
            return SparkW2V.load(cloud_path + model_name)
        except Exception as e:
            raise ValueError("load_vectors(): failed with exception!") from e

    def save_model(self, cloud_path: str, model_name: str,) -> None:
        """
        Saves the word2vec model object to a cloud bucket, always overwrite.
        """
        try:
            if not cloud_path.endswith("/"):
                cloud_path += "/"
            if not model_name.endswith(".model"):
                model_name += ".model"
            self.model.save(cloud_path + model_name)  # type: ignore
        except Exception as e:
            raise ValueError("save_model(): failed with exception!") from e

    def load_model(self, cloud_path: str, model_name: str,) -> SparkW2VModel:
        """
        Load a previously saved Word2Vec model object to memory.
        """
        try:
            if not cloud_path.endswith("/"):
                cloud_path += "/"
            if not model_name.endswith(".model"):
                model_name += ".model"
            return SparkW2VModel.load(cloud_path + model_name)
        except Exception as e:
            raise ValueError("load_model(): failed with exception!") from e
