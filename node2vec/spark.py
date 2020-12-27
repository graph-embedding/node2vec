import logging
import time
import random
from typing import Union
from typing import Any
from typing import Dict
from typing import Optional
from typing import List
from typing import Tuple
from typing import Set
from typing import Callable
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.sql.types import ArrayType
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import Row
from pyspark.sql import functions as ssf
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import Word2VecModel


# the default maximum out degrees of each vertex
MAX_OUT_DEGREES: int = 500000


# num of partitions for map partitions
NUM_PARTITIONS: int = 3000


# the default node2vec parameters
NODE2VEC_PARAMS: Dict[str, Any] = {
    #  num of walks starting from each node on sampling random walks, [10, 30]
    "num_walks": 20,
    #  length of each random walk path, [10, 30]
    "walk_length": 10,
    #  parameter p in the Node2Vec paper, (0, inf), weight on the probability of
    #  returning to a node coming from, having this higher tends the walks to be more
    #  like a BFS, having this very high (> 2) makes search very local
    "return_param": 1.0,
    #  parameter q in the Node2Vec paper, (0, inf), weight on the probability of
    #  visiting a neighbor node to the starting node, having this higher tends the
    #  walks to be more like a DFS, having this very low makes search very local
    "inout_param": 1.0,
}


# the default word2vec parameters
WORD2VEC_PARAMS: Dict[str, Any] = {
    # min token freq to be included in the word2vec model's vocabulary
    "minCount": 0,
    # num of partitions for sentences of words
    "numPartitions": 100,
    "stepSize": 0.025,
    "maxIter": 10,
    "seed": None,
    # max length (in words) of each sentence in the input data. Any sentence longer
    # than this threshold will be divided into chunks up to the size
    "maxSentenceLength": 10000,
    # num of neighborhood nodes (context [-window, window])
    "windowSize": 5,
    # num of dimensions of the output graph embedding representation, e.g. 64, 128, 256
    "vectorSize": 128,
}


#
def generate_alias_tables(
    node_weights: List[float],
) -> Tuple[List[int], List[float]]:
    """
    Generate the two utility table for the Alias Method, following the original
    node2vec code.

    :param node_weights: a list of neighboring nodes and their weights

    return the two utility tables as lists
        probs: the probability table holding the relative probability of each neighbor
        alias: the alias table holding the alias index to be sampled from
    """
    n = len(node_weights)
    alias = [0 for _ in range(n)]
    avg_weight = sum(node_weights) / n
    probs = [x / avg_weight for x in node_weights]

    underfull, overfull = [], []
    for i in range(n):
        if probs[i] < 1.0:
            underfull.append(i)
        else:
            overfull.append(i)

    while underfull and overfull:
        under, over = underfull.pop(), overfull.pop()
        alias[under] = over
        probs[over] = probs[over] + probs[under] - 1.0
        if probs[over] < 1.0:
            underfull.append(over)
        else:
            overfull.append(over)
    return alias, probs


#
def generate_edge_alias_tables(
    src_id: int,
    src_nbs_id: Set[int],
    dst_nbs: List[int],
    dst_nbs_wt: List[float],
    return_param: float = 1.0,
    inout_param: float = 1.0,
) -> Tuple[List[int], List[float]]:
    """
    Apply the biased sampling on edge weight described in the node2vec paper. Each entry
    here represents an edge, and the src and dst node's info.

    :param src_id: the source node id
    :param src_nbs_id: the intersection of src neighbor and dst neighbor id's
    :param dst_nbs: the list of destination node's neighbor node id's
    :param dst_nbs_wt: the list of destination node's neighbor node's weights
    :param return_param: the parameter p defined in the paper
    :param inout_param: the parameter q defined in the paper

    return the utility tables of the Alias method, after weights are biased.
    """
    if len(dst_nbs) != len(dst_nbs_wt):
        raise ValueError(f"Invalid neighbors tuple '{dst_nbs}'!")
    if return_param == 0 or inout_param == 0:
        raise ValueError(
            f"Zero return ({return_param}) or inout ({inout_param}) parameter!"
        )
    # apply bias to edge weights
    neighbors_dst: List[float] = []
    for i in range(len(dst_nbs)):
        dst_neighbor_id, weight = dst_nbs[i], dst_nbs_wt[i]
        # go back to the src id
        if dst_neighbor_id == src_id:
            unnorm_prob = weight / return_param
        # go to a neighbor of src
        elif dst_neighbor_id in src_nbs_id:
            unnorm_prob = weight
        # go to a brand new vertex
        else:
            unnorm_prob = weight / inout_param
        neighbors_dst.append(unnorm_prob)
    return generate_alias_tables(neighbors_dst)


#
def _sampling_from_alias_wiki(
    alias: List[int],
    probs: List[float],
    random_val: float,
) -> int:
    """
    Draw sample from a non-uniform discrete distribution using Alias sampling.
    This implementation is aligned with the wiki description using 1 random number.

    :param alias: the alias list in range [0, n)
    :param probs: the pseudo-probability table
    :param random_val: a random floating point number in the range [0.0, 1.0)

    Return the picked index in the neighbor list as next node in the random walk path.
    """
    n = len(alias)
    pick = int(n * random_val)
    y = n * random_val - pick
    if y < probs[pick]:
        return pick
    else:
        return alias[pick]


#
def _sampling_from_alias(
    alias: List[int],
    probs: List[float],
    first_random: float,
    second_random: float,
) -> int:
    """
    This is aligned with the original node2vec implementation w/ 2 random numbers.

    :param alias: the pre-calculated alias list
    :param probs: the pre-calculated probs list
    :param first_random: 1st random floating point number in the range [0.0, 1.0)
    :param second_random: 2nd random floating point number in the range [0.0, 1.0)

    Return the picked index in the neighbor list as next vertex in the random walk path.
    """
    pick = int(first_random * len(alias))
    if second_random < probs[pick]:
        return pick
    else:
        return alias[pick]


#
def extend_random_walk(
    path: List[int],
    dst_neighbors: List[int],
    alias: List[int],
    probs: List[float],
    first_random: float,
    second_random: Optional[float] = None,
) -> List[int]:
    """
    Extend the random walk path by making a biased random sampling at next step

    :param path: the existing random walk path
    :param dst_neighbors: the neighbor node id's of the dst node
    :param alias: the pre-calculated alias list
    :param probs: the pre-calculated probs list
    :param first_random: 1st random floating point number in the range [0.0, 1.0)
    :param second_random: 2nd random floating point number in the range [0.0, 1.0),
                          controlling which sampling method to be used.

    Return the extended path with one more node in the random walk path.
    """
    if second_random is not None:
        next_index = _sampling_from_alias(alias, probs, first_random, second_random)
    else:
        next_index = _sampling_from_alias_wiki(alias, probs, first_random)

    next_vertex = dst_neighbors[next_index]
    # first step
    if len(path) == 2 and path[0] < 0:
        path = [path[1], next_vertex]
    # remaining step
    else:
        path.append(next_vertex)
    return path


#
# ============================- transformer func ====================================
#
def trim_hotspot_vertices(
    max_out_degree: int = 0,
    random_seed: Optional[int] = None,
) -> Callable:
    """
    This func is to do random sampling on the edges of vertices which have very large
    number of out edges. A maximal threshold is provided and random sampling is applied.
    By default the threshold is 100,000.

    :param max_out_degree: the max out degree of each vertex to avoid hotspot
    :param random_seed: the seed for random sampling, testing only
    """
    if max_out_degree <= 0:
        max_out_degree = MAX_OUT_DEGREES
    if random_seed is not None:
        random.seed(random_seed)

    def _hotspot_vertices_trimming(partition: List[Row]) -> List[Row]:
        """
        :param partition: a partition in List[Row] of ["src", "dst", "weight"]
        return a List[Row] of shared neighbors of every pair of src and dst
        """
        src_nbs: Dict[str, List[Tuple[str, float]]] = {}
        for arow in partition:
            row = arow.asDict()
            src = row["src"]
            if src not in src_nbs:
                src_nbs[src] = []
            src_nbs[src].append((row["dst"], row["weight"]))

        result: List[Row] = []
        for src, nbs in src_nbs.items():
            if len(nbs) > max_out_degree:
                nbs = random.sample(nbs, max_out_degree)
            for dst, weight in nbs:
                result += [Row(src=src, dst=dst, weight=weight)]
        return result

    return _hotspot_vertices_trimming


def get_vertex_neighbors(partition: List[Row]) -> List[Row]:
    """
    Aggregate all neighbors and their weights for every vertex in the graph

    :param partition: a partition in List[Row] of the input edge dataframe
    Return a List[Row] with cols ["id", "neighbors", "weights"]
    """
    src_neighbors: Dict[int, List[Tuple[int, float]]] = {}
    for arow in partition:
        row = arow.asDict()
        src = row["src"]
        if src not in src_neighbors:
            src_neighbors[src] = []
        src_neighbors[src].append((row["dst"], row["weight"]))

    result: List[Row] = []
    for src, neighbors in src_neighbors.items():
        neighbors.sort(key=lambda x: x[0])
        nbs, wts = zip(*neighbors)
        result += [Row(id=src, neighbors=list(nbs), weights=list(wts))]
    return result


#
def initiate_random_walk(num_walks: int) -> Callable:
    """
    Initiate the random walk path and replicate for num_walks times

    :param num_walks: the num of random walks starting from each vertex
    """

    def _get_random_walk_started(partition: List[Row]) -> List[Row]:
        """
        :param partition: a partition in List[Row] of single-col ["dst"]
        return a List[Row] of random walks of every src vertex
        """
        result: List[Row] = []
        for arow in partition:
            row = arow.asDict()
            dst = row["id"]
            for i in range(1, num_walks + 1):
                result += [Row(src=-i, dst=dst, path=[-i, dst])]
        return result

    return _get_random_walk_started


# a transformer func for mapPartitionsWithIndex(index, partition)
def next_step_random_walk(
    return_param: float,
    inout_param: float,
    random_seed: Optional[int] = None,
) -> Callable:
    """
    Extend the random walk path by one more step

    :param return_param: the p parameter
    :param inout_param: the q parameter
    :param random_seed: optional random seed, for testing only
    """

    def _get_random_next_step(index: int, partition: List[Row]) -> List[Row]:
        """
        :param index: the partition index
        :param partition: a partition in List[Row] with cols ["src", "dst", "path",
                          "src_neighbors", "dst_neighbors"]
        return a List[Row] with cols ["src", "dst", "path"]
        """
        if random_seed is not None:
            random.seed(random_seed + index * 100)
        result: List[Row] = []
        for arow in partition:
            row = arow.asDict()
            src, src_nbs = row["src"], row["src_neighbors"]
            src_nbs_id = set() if src_nbs is None else set(src_nbs)
            dst_nbs, dst_nbs_wt = row["neighbors"], row["weights"]
            if src < 0:
                alias, probs = generate_alias_tables(dst_nbs_wt)
            else:
                alias, probs = generate_edge_alias_tables(
                    src,
                    src_nbs_id,
                    dst_nbs,
                    dst_nbs_wt,
                    return_param,
                    inout_param,
                )
            path = extend_random_walk(
                path=row["path"],
                dst_neighbors=dst_nbs,
                alias=alias,
                probs=probs,
                first_random=random.random(),
                second_random=random.random(),
            )
            result += [Row(src=path[-2], dst=path[-1], path=path)]
        return result

    return _get_random_next_step


#
def get_standard_paths(partition: List[Row]) -> List[Row]:
    """
    convert a random path from a list to a pair [src, walk]

    :param partition: a partition in List[Row] with cols ["src", "dst", "path"]
    return a List[Row] after calculating attributes as a partition ["src", "path"]
    """
    result: List[Row] = []
    for arow in partition:
        row = arow.asDict()
        path = row["path"]
        result += [Row(id=path[0], walk=path)]
    return result


#
# ============================- Node2Vec Core ====================================
#
class Node2VecSpark:
    """
    A wrapper class to handle input and output data for distributed node2vec algorithm.
    """

    def __init__(
        self,
        spark: SparkSession,
        n2v_params: Dict[str, Any],
        w2v_params: Dict[str, Any],
        max_out_degree: int = 0,
        df_users: Optional[DataFrame] = None,
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
        :param max_out_degree: int, the max num of edges allowed for each vertex
        :param df_users: a single-column df to refine random walk for selected users
        :param window_size: the context size for word2vec embedding
        :param vector_size: int, dimension of the output graph embedding representation
                            num of codes after transforming from words (dimension
                            of embedding feature representation), usually power of 2,
                            e.g. 64, 128, 256
        :param w2v_params: dict of parameters to pass to gensim's word2vec module (not
                           to set embedding dim here)
        :param random_seed: optional random seed, for testing only
        """
        self.spark = spark
        self.random_seed = random_seed if random_seed else int(time.time())
        self.df: Optional[DataFrame] = None
        self.name_id: Optional[DataFrame] = None
        self.df_adj: Optional[DataFrame] = None
        self.model: Optional[Word2VecModel] = None
        self.max_out_degree = max_out_degree
        self.users = None if df_users is None else df_users.toDF("id")

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
        self,
        df_graph: DataFrame,
        indexed: bool,
        directed: bool,
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

        # trim hotspot vertices
        df_graph = df_graph.withColumn("weight", df_graph["weight"].cast("float"))
        if directed is False:
            df_rev = df_graph.select("dst", "src", "weight")
            df_graph = df_graph.union(df_rev).distinct()
        df = df_graph.repartition(NUM_PARTITIONS, ["src"])
        param = {"max_out_degree": self.max_out_degree, "random_seed": self.random_seed}
        df_graph = self.spark.createDataFrame(
            df.rdd.mapPartitions(trim_hotspot_vertices(**param))
        ).cache()
        logging.info(f"Num of edges after trimming = {df_graph.count()}")
        if indexed is True:
            df = df_graph.select("src", "dst", "weight")
            df = df.withColumn("src", df["src"].cast("int"))
            self.df = df.withColumn("dst", df["dst"].cast("int"))
            return

        # index the input graph
        df = df_graph.select("src").union(df_graph.select("dst")).distinct().sort("src")
        name_id = df.rdd.zipWithIndex().map(lambda x: Row(name=x[0][0], id=int(x[1])))
        self.name_id = name_id.toDF().cache()
        logging.info(f"Num of indexed vertices: {self.name_id.count()}")
        df = df_graph.select("src", "dst", "weight").toDF("src1", "dst1", "weight")
        srcid = self.name_id.select("name", "id").toDF("src1", "src")
        dstid = self.name_id.select("name", "id").toDF("dst1", "dst")
        df_edge = df.join(srcid, on=["src1"]).join(dstid, on=["dst1"])
        self.df = df_edge.select("src", "dst", "weight").cache()
        logging.info(f"Num of indexed edges: {self.df.count()}")

        # process vertices for adjacency lists
        df = self.df.repartition(NUM_PARTITIONS, ["src"])
        self.df_adj = self.spark.createDataFrame(
            df.rdd.mapPartitions(get_vertex_neighbors),
            schema=StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("neighbors", ArrayType(IntegerType()), False),
                    StructField("weights", ArrayType(FloatType()), False),
                ]
            ),
        ).cache()
        logging.info(f"df_adj length = {self.df_adj.count()}")

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
        if self.df is None or self.name_id is None or self.df_adj is None:
            raise ValueError("Please validate and/or index the input graph")
        self.spark.sparkContext.setCheckpointDir(
            "gs://sq-dataproc-graphlib-prod/tmp/checkpoints",
        )

        # initiate random walk
        schema = StructType(
            [
                StructField("src", IntegerType(), False),
                StructField("dst", IntegerType(), False),
                StructField("path", ArrayType(IntegerType()), False),
            ]
        )
        num_walks = self.n2v_params["num_walks"]
        df_user = self.df_adj.select("id")
        if self.users is not None:
            df_user = df_user.join(self.users, on=["id"])
        walks = self.spark.createDataFrame(
            df_user.rdd.mapPartitions(initiate_random_walk(num_walks)),
            schema=schema,
        ).cache()
        logging.info(f"random_walk(): init walks length = {walks.count()}")

        # random walk with distributed bfs
        df_src = self.df_adj.select("id", "neighbors").toDF("src", "src_neighbors")
        df_dst = self.df_adj.withColumnRenamed("id", "dst")
        param_p = self.n2v_params["return_param"]
        param_q = self.n2v_params["inout_param"]
        for i in range(self.n2v_params["walk_length"]):
            next_walks = walks.join(df_src, on=["src"], how="left_outer")
            next_walks = next_walks.join(df_dst, on=["dst"]).drop("dst")
            walks = self.spark.createDataFrame(
                next_walks.rdd.mapPartitionsWithIndex(
                    next_step_random_walk(param_p, param_q, self.random_seed)
                ),
                schema=schema,
            )
            walks = walks.cache() if i % 10 < 9 else walks.checkpoint()
            logging.info(f"random_walk(): step {i} walks length = {walks.count()}")

        # convert paths back to lists
        df_walks = self.spark.createDataFrame(
            walks.rdd.mapPartitions(get_standard_paths),
            schema=StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("walk", ArrayType(IntegerType()), False),
                ]
            ),
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
            name_id = self.name_id.select("id", "name").toDF("word", "token")
            df = df.join(name_id, on=["word"]).select("token", "vector")
        return df

    def get_vector(self, vertex_name: Union[str, int]) -> DataFrame:
        """
        :param vertex_name: the vertex name
        Return the vector associated with a node or all vectors
        """
        if isinstance(vertex_name, int):
            vertex_name = str(vertex_name)
        return self.model.getVectors().filter(f"word = {vertex_name}")  # type: ignore

    def save_model(
        self,
        cloud_path: str,
        model_name: str,
    ) -> None:
        """
        Saves the word2vec model object to a cloud bucket, always overwrite.
        """
        if not model_name.endswith(".sparkml"):
            model_name += ".sparkml"
        self.model.save(cloud_path + "/" + model_name)  # type: ignore

    def load_model(
        self,
        cloud_path: str,
        model_name: str,
    ) -> Word2VecModel:
        """
        Load a previously saved Word2Vec model object to memory.
        """
        if not model_name.endswith(".sparkml"):
            model_name += ".sparkml"
        self.model = Word2VecModel.load(cloud_path + "/" + model_name)
        return self.model
