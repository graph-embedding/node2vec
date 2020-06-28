import logging
import time
from typing import Union
from typing import Any
from typing import Dict
from typing import Optional
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import Row
from pyspark.sql import functions as ssf
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import Word2VecModel

from node2vec.spark.constants import NUM_PARTITIONS
from node2vec.spark.constants import NODE2VEC_PARAMS
from node2vec.spark.constants import WORD2VEC_PARAMS
from node2vec.spark.utils import get_vertex_neighbors
from node2vec.spark.utils import get_edge_shared_neighbors
from node2vec.spark.utils import initiate_random_walk
from node2vec.spark.utils import next_step_random_walk
from node2vec.spark.utils import get_standard_paths


class Node2VecSpark:
    """
    A wrapper class to handle input and output data for distributed node2vec algorithm.
    """

    def __init__(
        self,
        spark: SparkSession,
        n2v_params: Dict[str, Any],
        w2v_params: Dict[str, Any],
        window_size: Optional[int] = None,
        vector_size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        A driver class for the distributed Node2Vec algorithm for vertex indexing,
        graph embedding, read and write vectors and/or models.

        :param spark: an active Spark session
        :param n2v_params: dict of the node2vec params for num_walks, walk_length,
                           return_param, and inout_param, with default values
        :param window_size: the context size for word2vec embedding
        :param vector_size: int, dimension of the output graph embedding representation
                            num of codes after transforming from words (dimension
                            of embedding feature representation), usually power of 2,
                            e.g. 64, 128, 256
        :param w2v_params: dict of parameters to pass to gensim's word2vec module (not
                           to set embedding dim here)
        :param random_seed: optional random seed, for testing only
        """
        logging.info("__init__(): preprocssing input graph data ...")
        self.spark = spark
        self.random_seed = random_seed if random_seed else int(time.time())
        self.df: Optional[DataFrame] = None
        self.name_id: Optional[DataFrame] = None
        self.model: Optional[Word2VecModel] = None

        # update n2v_params
        for param in NODE2VEC_PARAMS:
            if param not in n2v_params:
                n2v_params[param] = NODE2VEC_PARAMS[param]
        self.n2v_params = n2v_params

        # update w2v_params
        for param in WORD2VEC_PARAMS:
            if param not in w2v_params:
                w2v_params[param] = WORD2VEC_PARAMS[param]
        w2v_params["seed"] = self.random_seed
        if window_size is not None:
            if window_size < 5 or window_size > 30:
                raise ValueError(f"Inappropriate context window size {window_size}!")
            w2v_params["windowSize"] = window_size
        if vector_size is not None:
            if vector_size < 32 or vector_size > 1024:
                raise ValueError(f"Inappropriate vector dimension {vector_size}!")
            w2v_params["vectorSize"] = vector_size
        self.w2v_params = w2v_params
        logging.info(f"__init__(): w2v params: {w2v_params}")
        self.word2vec = Word2Vec(inputCol="walk", outputCol="vector", **w2v_params)

    def preprocess_input_graph(
        self, df_graph: DataFrame, indexed: bool, directed: bool,
    ) -> None:
        """
        Preprocess the input graph dataframe so that it returns a dataframe in
        standard format ['src', 'dst', 'weight']

        If required, index all vertices and edges in a graph by using an int32 to
        represent a vertex

        :param df_graph: a Spark dataframe w/ cols "src", "dst", and optional "weight"
        :param indexed: bool, True if the input graph has been indexed, or False
        :param directed: bool, directed graph or not

        Return the validated and indexed (if needed) dataframe or throw exceptions
        """
        if "src" not in df_graph.columns or "dst" not in df_graph.columns:
            raise ValueError(f"Input graph NOT in the right format: {df_graph.columns}")
        if "weight" not in df_graph.columns:
            df_graph = df_graph.withColumn("weight", ssf.lit(1.0))
        df_graph = df_graph.withColumn("weight", df_graph["weight"].cast("float"))
        if directed is False:
            df_rev = df_graph.select("dst", "src", "weight")
            df_graph = df_graph.union(df_rev)

        if indexed is True:
            self.df = (
                df_graph.select("src", "dst", "weight")
                .withColumn("src", df_graph["src"].cast("int"))
                .withColumn("dst", df_graph["dst"].cast("int"))
            )
            return
        # index the input graph
        df = df_graph.select("src").union(df_graph.select("dst")).distinct().sort("src")
        name_id = (
            df.rdd.zipWithIndex().map(lambda x: Row(name=x[0][0], id=int(x[1]))).toDF()
        ).cache()
        self.name_id = name_id
        logging.info(f"Num of indexed vertices: {self.name_id.count()}")

        df = df_graph.withColumnRenamed("src", "src1").withColumnRenamed("dst", "dst1")
        srcid = name_id.withColumnRenamed("name", "src1").withColumnRenamed("id", "src")
        dstid = name_id.withColumnRenamed("name", "dst1").withColumnRenamed("id", "dst")
        df_edge = df.join(srcid, on=["src1"]).join(dstid, on=["dst1"])
        self.df = df_edge.select("src", "dst", "weight").cache()
        logging.info(f"Num of indexed edges: {self.df.count()}")

    def random_walk(self) -> DataFrame:
        """
        A simulated random walk process. Current implementation is naive since it uses
        two for loops and hence could take relatively long time. A better way is to
        duplicate the graph by "num_walks" times, and in each copy there is a different
        random seed, and all these copies run parallel in distributed nodes. However,
        this implementation requires much more memory (num_walks x graph_size).

        It will takes four important tunable parameters specified in the paper, and
        they should be considered as hyperparameters. An optimal graph embedding can
        only be achieved with optimal hyperparameters. This HP tuning requires a lot
        more computing resources. Overall, this "node2vec" implementation is very
        computing and memory intensive.

        Returns a two-column DataFrame ["src", "walk"], where "src" is the source
            vertex id, and "walk" is a random walk path as a list of vertex id's
        ------------------
        src   walk
        ------------------
        1     [1 3 5 5 2]
        3     [3 2 4 7 1]
        7     [7 8 1 7 4]
        ------------------
        """
        if self.df is None:
            raise ValueError("Please validate and/or index the input graph")
        logging.info("random_walk(): start random walking ...")
        # process vertices
        df = self.df.repartition(NUM_PARTITIONS, ["src"])
        df_dst = self.spark.createDataFrame(
            df.rdd.mapPartitions(get_vertex_neighbors)
        ).cache()
        logging.info(f"random_walk(): df_dst length = {df_dst.count()}")

        # process edges
        num_walks = self.n2v_params["num_walks"]
        df = (
            self.df.select("src", "dst")
            .join(df_dst, on=["dst"])
            .repartition(NUM_PARTITIONS, ["src"])
        )
        df_edge = self.spark.createDataFrame(
            df.rdd.mapPartitions(get_edge_shared_neighbors(num_walks))
        ).cache()
        logging.info(f"random_walk(): df_edge length = {df_edge.count()}")

        # conduct random walk with distributed bfs
        walks = self.spark.createDataFrame(
            df_dst.select("dst").rdd.mapPartitions(initiate_random_walk(num_walks))
        ).cache()
        logging.info(f"random_walk(): init walks length = {walks.count()}")
        param_p = self.n2v_params["return_param"]
        param_q = self.n2v_params["inout_param"]
        for i in range(self.n2v_params["walk_length"]):
            next_walks = walks.join(df_dst, on=["dst"]).join(df_edge, on=["src", "dst"])
            walks_rdd = next_walks.rdd.mapPartitionsWithIndex(
                next_step_random_walk(param_p, param_q, self.random_seed)
            )
            next_walks = self.spark.createDataFrame(walks_rdd).cache()
            logging.info(
                f"random_walk(): round {i} walks length = {next_walks.count()}"
            )
            walks.rdd.unpersist()
            walks = next_walks

        # convert paths back to lists
        df_walks = self.spark.createDataFrame(
            walks.rdd.mapPartitions(get_standard_paths)
        ).cache()
        logging.info(f"random_walk(): num of walks generated: {df_walks.count()}")
        return df_walks

    def fit(self, df_walks: Optional[DataFrame] = None) -> Word2VecModel:
        """
        the entry point for fitting a node2vec process for graph feature embedding

        :param df_walks: the DataFrame of random walks: ["src", "walk"], where "walk"
                         is a list column
        return the fit model of Word2Vec
        """
        if df_walks is None:
            raise ValueError("Please conduct random walk on the input graph")
        df_walks = df_walks.select("walk").withColumn(
            "walk", df_walks["walk"].cast("array<string>")
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
        df = self.model.getVectors()
        if self.name_id is not None:
            df = df.join(self.name_id, on=["id"])
            df = df.select(df["name"].alias("token"), "vector")
        return df

    def get_vector(self, vertex_name: Union[str, int]) -> DataFrame:
        """
        :param vertex_name: the vertex name
        Return the vector associated with a node or all vectors
        """
        if isinstance(vertex_name, int):
            vertex_name = str(vertex_name)
        return self.model.getVectors().filter(f"word = {vertex_name}")  # type: ignore

    def save_model(self, cloud_path: str, model_name: str,) -> None:
        """
        Saves the word2vec model object to a cloud bucket, always overwrite.
        """
        if not model_name.endswith(".sparkml"):
            model_name += ".sparkml"
        self.model.save(cloud_path + "/" + model_name)  # type: ignore

    def load_model(self, cloud_path: str, model_name: str,) -> Word2VecModel:
        """
        Load a previously saved Word2Vec model object to memory.
        """
        if not model_name.endswith(".sparkml"):
            model_name += ".sparkml"
        self.model = Word2VecModel.load(cloud_path + "/" + model_name)
        return self.model
