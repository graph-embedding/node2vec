import networkx as nx
import logging
import time
from typing import Union
from typing import List
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
from node2vec.spark.constants import WORD2VEC_PARAMS
from node2vec.spark.utils import next_step_random_walk
from node2vec.spark.utils import aggregate_vertex_neighbors
from node2vec.spark.utils import calculate_vertex_attributes
from node2vec.spark.utils import calculate_edge_attributes


class Node2VecSpark:
    """
    A wrapper class to handle input and output data for distributed node2vec algorithm.
    """

    def __init__(
        self,
        spark: SparkSession,
        input_graph: Union[DataFrame, nx.Graph],
        indexed: bool = False,
        directed: bool = True,
        num_walks: int = 30,
        walk_length: int = 20,
        return_param: float = 1.0,
        inout_param: float = 1.0,
        vector_size: int = 128,
        w2vparams: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        A driver class for the distributed Node2Vec algorithm, for node indexing,
        graph embedding, read and write vectors and/or models.

        :param spark: a Spark session
        :param input_graph: data for constructing the graph, either a Spark DataFrame of
                            edge lists or a Networkx Graph object (for synthetic graphs)
        :param indexed: bool, True if the input graph has been indexed, or False
        :param directed: bool, directed graph or not

        :param num_walks: int, num of walks starting from each node on sampling random
                          walks, usually [10, 30], tunable
        :param walk_length: int, length of each random walk, usually [10, 30], tunable
        :param return_param: float in (0, inf), parameter p in the Node2Vec paper,
            weight on the probability of returning to a node coming from
            Having this higher tends the walks to be more like a Breadth-First Search.
            Having this very high (> 2) makes search very local. tunable
        :param inout_param: float in (0, inf), parameter q in the Node2Vec paper
            the weight on the probability of visiting a neighbor node to the one we're
            coming from in the random walk
            Having this higher tends the walks to be more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.  tunable

        :param vector_size: int, dimension of the output graph embeding representation
                            num of codes after transforming from words (dimension
                            of embedding feature representation), usually power of 2,
                            e.g. 64, 128, 256
        :param w2vparams: dict of parameters to pass to gensim's word2vec module (not
                          to set embedding dim here)
        :param random_seed: int, the global random sampling seed
        """
        logging.info("__init__(): preprocssing input graph data ...")
        self.spark = spark
        self.walks: Optional[Word2VecModel] = None
        self.model: Optional[Word2VecModel] = None
        self.vertices: Optional[DataFrame] = None
        self.edges: Optional[DataFrame] = None

        # index the graph and construct mapping from node name to node id
        df_graph = self._validate_input_graph(spark, input_graph)
        if directed is False:
            df_rev = df_graph.select("dst", "src", "weight")
            df_graph = df_graph.union(df_rev)
        # the format-compliant indexed graph
        self.df = self._index_graph(spark, df_graph) if indexed is False else df_graph

        # word2vec: params and model
        if vector_size < 32 or vector_size > 1024:
            raise ValueError(f"Inappropriate vector dimension {vector_size}!")
        if num_walks < 2 or walk_length < 2:
            raise ValueError(f"Error as num_walks='{num_walks}' length='{walk_length}'")
        if w2vparams is None:
            w2vparams = WORD2VEC_PARAMS

        self.random_seed = random_seed or int(round(time.time() * 1000))
        if w2vparams.get("seed", None) is None:
            w2vparams["seed"] = self.random_seed
        logging.info(f"__init__(): w2v params: {w2vparams}")
        self.word2vec = Word2Vec(
            vectorSize=vector_size, inputCol="walk", outputCol="vector", **w2vparams,
        )
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.return_param = return_param
        self.inout_param = inout_param

    @staticmethod
    def _validate_input_graph(
        spark: SparkSession, input_graph: Union[DataFrame, nx.Graph],
    ) -> DataFrame:
        """
        a helper func to preprocess the input graph dataframe so that it returns a
        dataframe in standard format ['src', 'dst', 'weight']

        :param spark: an active SparkSession
        :param input_graph: the Spark Dataframe of edge lists or networkx graph object

        return the processed dataframe or throw exceptions
        """
        try:
            if isinstance(input_graph, nx.Graph):
                graph = input_graph.edges.data("weight", default=1.0)
                edge_list = [{"src": e[0], "dst": e[1], "weight": e[2]} for e in graph]
                df_graph = spark.createDataFrame(Row(**r) for r in edge_list).select(
                    "src", "dst", "weight"
                )
            else:
                if "weight" not in input_graph.columns:
                    input_graph = input_graph.withColumn("weight", ssf.lit(1.0))
                df_graph = input_graph.select("src", "dst", "weight")
                df_graph = (
                    df_graph.withColumn("src", df_graph["src"].cast("int"))
                    .withColumn("dst", df_graph["dst"].cast("int"))
                    .withColumn("weight", df_graph["weight"].cast("float"))
                )
        except Exception as e:
            raise ValueError("Input graph is in wrong format!") from e
        return df_graph

    @staticmethod
    def _index_graph(spark: SparkSession, df: DataFrame,) -> DataFrame:
        """
        Index all vertices and edges in a graph by using an int32 to represent a vertex

        :param spark: an active SparkSession
        :param df: the Spark Dataframe of edge lists
        return the indexed DataFrame with vertex id and vertex name columns.
        """
        df_node = df.select("src").union(df.select("dst")).distinct()
        df_node.createOrReplaceTempView("df_node")
        name_id = spark.sql(
            """
            SELECT src AS vertex_name,
                   row_number() OVER(ORDER BY src) - 1 AS vertex_id
              FROM df_node
        """
        ).cache()
        logging.info(f"Total number of vertices: {name_id.count()}")

        df.createOrReplaceTempView("df")
        name_id.createOrReplaceTempView("name_id")
        df_graph = spark.sql(
            """
            SELECT a.vertex_id AS src,
                   a.vertex_name AS name,
                   b.vertex_id AS dst,
                   df.weight AS weight
              FROM df
              JOIN name_id AS a
                ON df.src = a.vertex_name
              JOIN name_id AS b
                ON df.dst = b.vertex_name
        """
        ).cache()
        logging.info(f"size of indexed graph: {df_graph.count()}")
        return df_graph

    def init_node_dataframe(self) -> DataFrame:
        """
        A utility function to initiate the state of the Node2Vec model:
        1) normalize edge weights;
        2) initiate the transition probabilities
        3) create vertices and edges dataframe for a GraphFrame

        Returns the node dataframe with two node attributes:
            neighbors: List[Tuple[int, float]] = []
            path: List[int] = []
        where each vertex replicates for "num_walks" times
        """
        vertex = self.df.repartition(NUM_PARTITIONS, ["src"]).rdd.mapPartitions(
            calculate_vertex_attributes(self.num_walks)
        )
        self.vertices = self.spark.createDataFrame(vertex).cache()
        logging.info(f"number of walks to be generated: {self.vertices.count()}")
        return self.vertices

    def init_edge_dataframe(self) -> DataFrame:
        """
        A utility function to initiate the state of the Node2Vec model:
        1) normalize edge weights;
        2) initiate the transition probabilities
        3) create vertices and edges dataframe for a GraphFrame

        Returns the edge dataframe with three edge attributes:
             dst_neighbors: List[int] = []
             alias: List[int] = []
             probs: List[float] = []
        """
        rdd_vertex = self.df.repartition(NUM_PARTITIONS, ["src"]).rdd.mapPartitions(
            aggregate_vertex_neighbors
        )
        df_vertex = self.spark.createDataFrame(rdd_vertex).cache()
        logging.info(f"number of vertices: {df_vertex.count()}")

        df_vertex.createOrReplaceTempView("df_vertex")
        self.df.repartition(NUM_PARTITIONS).createOrReplaceTempView("edges")
        edges = self.spark.sql(
            """
            SELECT edges.src,
                   edges.dst,
                   a.neighbors AS src_neighbors,
                   b.neighbors AS dst_neighbors
              FROM edges
              JOIN df_vertex AS a
                ON edges.src = a.id
              JOIN df_vertex AS b
                ON edges.dst = b.id
        """
        ).repartition(NUM_PARTITIONS, ["src"])

        self.edges = (
            self.spark.createDataFrame(
                edges.rdd.mapPartitions(
                    calculate_edge_attributes(
                        self.return_param, self.inout_param, self.num_walks,
                    )
                )
            )
            .select("edge", "dst_neighbors", "alias", "probs")
            .cache()
        )
        df_vertex.unpersist()
        return self.edges

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
        if self.edges is None:
            self.edges = self.init_edge_dataframe()
        rdd_edge = self.edges.rdd.map(
            lambda x: (x["edge"], (x["dst_neighbors"], x["alias"], x["probs"]))
        )
        if self.vertices is None:
            self.vertices = self.init_node_dataframe()
        df_path = self.vertices.select("id", "path")
        for i in range(self.walk_length):
            rand_seed = self.random_seed + i * 1000
            df_path = (
                df_path.withColumn("rand", ssf.rand(rand_seed))
                .rdd.map(lambda x: (f"{x['path'][-2]} {x['path'][-1]}", x))
                .join(rdd_edge)
                .map(
                    lambda x: (
                        x[1][0]["id"],  # src id
                        next_step_random_walk(
                            x[1][0]["path"],  # path
                            x[1][0]["rand"],  # random double
                            x[1][1][0],  # dst_neighbors
                            x[1][1][1],  # jvar
                            x[1][1][2],
                        ),  # qvar
                    )
                )
                .toDF(["id", "path"])
            )
        self.walks = (
            df_path.withColumnRenamed("id", "src")
            .withColumnRenamed("path", "walk")
            .repartition(NUM_PARTITIONS)
        )
        return self.walks

    def fit(self, df_walks: Optional[DataFrame] = None) -> Word2VecModel:
        """
        the entry point for fitting a node2vec process for graph feature embedding

        :param df_walks: the DataFrame of random walks: ["src", "walk"], where "walk"
                         is a list column
        return the fit model of Word2Vec
        """
        if df_walks is None:
            df_walks = self.random_walk()
        df_walks = df_walks.select("walk").withColumn(
            "walk", df_walks["walk"].cast("array<string>")
        )
        self.model = self.word2vec.fit(df_walks)
        logging.info("model fitting done!")
        return self.model

    def embedding(self, df_walks: Optional[DataFrame] = None) -> DataFrame:
        """
        the entry point for fitting a node2vec process of graph embedding

        :param df_walks: the DataFrame of random walks: ["src", "walk"], where "walk"
                         is a list column
        Returns the resulting embedding vectors as a dataframe
        """
        self.model = self.fit(df_walks)
        return self.model.getVectors()

    def get_vector(self, vertex_name: Union[str, int]) -> List[float]:
        """
        vertex_name: str or int, either the node ID or name depending on graph format

        Return vector associated with a node identified by the original node name/id
        """
        if isinstance(vertex_name, str):
            if "name" not in self.vertices.columns:  # type: ignore
                raise ValueError(f"Input vertex name '{vertex_name}' is NOT indexed!")
            vertex = self.vertices.filter(f"name = '{vertex_name}'")  # type: ignore
            vertex_id = int(vertex.collect()[0])
        else:
            vertex_id = vertex_name
        try:
            vec = self.model.getVectors().filter(f"word = {vertex_id}")  # type: ignore
            res = vec.select("vector").head().asDict()
            return list(res["vector"])
        except Exception as e:
            raise ValueError(f"Failed to get vector for {vertex_name}!") from e

    @staticmethod
    def _validate_path(cloud_path: str) -> str:
        if not cloud_path.endswith("/"):
            cloud_path += "/"
        return cloud_path

    def save_vectors(self, cloud_path: str, model_name: str,) -> None:
        """
        Save as graph embedding vectors as a Spark DataFrame

        :param cloud_path: a gcs or s3 bucket
        :param model_name: the name to be used for the model file
        """
        cloud_path = self._validate_path(cloud_path)
        try:
            self.word2vec.save(cloud_path + model_name)
        except Exception as e:
            raise ValueError("save_vectors(): failed with exception!") from e

    def load_vectors(self, cloud_path: str, model_name: str,) -> DataFrame:
        """
        Load graph embedding vectors from saved file to a Spark DataFrame

        :param cloud_path: a gcs or s3 bucket
        :param model_name: the name to be used for the model file
        returns a dataframe
        """
        cloud_path = self._validate_path(cloud_path)
        try:
            vectors = Word2Vec.load(cloud_path + model_name)
            return vectors
        except Exception as e:
            raise ValueError("load_vectors(): failed with exception!") from e

    @staticmethod
    def _validate_modelname(model_name: str) -> str:
        if not model_name.endswith(".model"):
            model_name += ".model"
        return model_name

    def save_model(self, cloud_path: str, model_name: str,) -> None:
        """
        Saves the word2vec model object to a cloud bucket, always overwrite.

        :param cloud_path: a gcs or s3 bucket
        :param model_name: the name to be used for the model file
        """
        cloud_path = self._validate_path(cloud_path)
        model_name = self._validate_modelname(model_name)
        try:
            self.model.save(cloud_path + model_name)  # type: ignore
        except Exception as e:
            raise ValueError("save_model(): failed with exception!") from e

    def load_model(self, cloud_path: str, model_name: str,) -> Word2VecModel:
        """
        Load a previously saved Word2Vec model object to memory.

        :param cloud_path: a gcs or s3 bucket
        :param model_name: the name to be used for the model file, append ".model" to
                           it if it doesn't end with ".model".
        """
        cloud_path = self._validate_path(cloud_path)
        model_name = self._validate_modelname(model_name)
        try:
            model = Word2VecModel.load(cloud_path + model_name)
            return model
        except Exception as e:
            raise ValueError("load_model(): failed with exception!") from e
