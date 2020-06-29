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
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import Row
from pyspark.sql import functions as ssf
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import Word2VecModel

from node2vec.constants import NODE2VEC_PARAMS
from node2vec.constants import WORD2VEC_PARAMS


# num of partitions for map partitions
NUM_PARTITIONS: int = 3000


#
def generate_alias_tables(node_weights: List[float],) -> Tuple[List[int], List[float]]:
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
    shared_neighbor_ids: List[int],
    dst_neighbors: Tuple[List[int], List[float]],
    return_param: float = 1.0,
    inout_param: float = 1.0,
) -> Tuple[List[int], List[float]]:
    """
    Apply the biased sampling on edge weight described in the node2vec paper. Each entry
    here represents an edge, and the src and dst node's info.

    :param src_id: the source node id
    :param shared_neighbor_ids: the intersection of src neighbor and dst neighbor id's
    :param dst_neighbors: the list of destination node's neighbor node id's and weights
    :param return_param: the parameter p defined in the paper
    :param inout_param: the parameter q defined in the paper

    return the utility tables of the Alias method, after weights are biased.
    """
    if len(dst_neighbors) != 2 or len(dst_neighbors[0]) != len(dst_neighbors[1]):
        raise ValueError(f"Invalid neighbors tuple '{dst_neighbors}'!")
    if return_param == 0 or inout_param == 0:
        raise ValueError(
            f"Zero return ({return_param}) or inout ({inout_param}) parameter!"
        )
    # apply bias to edge weights
    shared_nb_ids = set(shared_neighbor_ids)
    neighbors_dst: List[float] = []
    for i in range(len(dst_neighbors[0])):
        dst_neighbor_id, weight = dst_neighbors[0][i], dst_neighbors[1][i]
        # go back to the src id
        if dst_neighbor_id == src_id:
            unnorm_prob = weight / return_param
        # go to a neighbor of src
        elif dst_neighbor_id in shared_nb_ids:
            unnorm_prob = weight
        # go to a brand new vertex
        else:
            unnorm_prob = weight / inout_param
        neighbors_dst.append(unnorm_prob)
    return generate_alias_tables(neighbors_dst)


#
def _sampling_from_alias_wiki(
    alias: List[int], probs: List[float], random_val: float,
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
    alias: List[int], probs: List[float], first_random: float, second_random: float,
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
def get_vertex_neighbors(partition: List[Row]) -> List[Row]:
    """
    Aggregate all neighbors and their weights for every vertex in the graph

    :param partition: a partition in List[Row] of the input edge dataframe
    Return a List[Row] with cols ["dst", "dst_neighbors"]
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
        result += [Row(dst=src, dst_neighbors=neighbors)]
    return result


#
def get_edge_shared_neighbors(num_walks: int) -> Callable:
    """
    Get the shared neighbors of the src and dst vertex of every edge in the graph

    :param num_walks: the num of random walks starting from each vertex
    """

    def _get_src_dst_shared_neighbors(partition: List[Row]) -> List[Row]:
        """
        :param partition: a partition in List[Row] of ["src", "dst", "dst_neighbors"]
        return a List[Row] of shared neighbors of every pair of src and dst
        """
        src_nbs: Dict[int, Set[int]] = {}
        src_dst_nbs: Dict[int, Dict[int, List[int]]] = {}
        for arow in partition:
            row = arow.asDict()
            src, dst = row["src"], row["dst"]
            if src not in src_nbs:
                src_nbs[src] = set()
            src_nbs[src].add(dst)
            if src not in src_dst_nbs:
                src_dst_nbs[src] = {}
            src_dst_nbs[src][dst] = [int(x) for x, _ in row["dst_neighbors"]]

        result: List[Row] = []
        for src, nbs_id in src_nbs.items():
            for i in range(1, num_walks + 1):
                result += [Row(src=-i, dst=src, shared_neighbor_ids=[])]
            for dst in nbs_id:
                shared_ids = [x for x in src_dst_nbs[src][dst] if x in nbs_id]
                result += [Row(src=src, dst=dst, shared_neighbor_ids=shared_ids)]
        return result

    return _get_src_dst_shared_neighbors


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
            dst = row["dst"]
            for i in range(1, num_walks + 1):
                result += [Row(src=-i, dst=dst, path=[-i, dst])]
        return result

    return _get_random_walk_started


# a transformer func for mapPartitionsWithIndex(index, partition)
def next_step_random_walk(
    return_param: float, inout_param: float, random_seed: Optional[int] = None,
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
                          "dst_neighbors", "shared_neighbor_ids"]
        return a List[Row] with cols ["src", "dst", "path"]
        """
        if random_seed is not None:
            random.seed(random_seed + index * 100)
        result: List[Row] = []
        for arow in partition:
            row = arow.asDict()
            src = row["src"]
            dst_nbs_id, dst_nbs_wt = zip(*row["dst_neighbors"])
            if src < 0:
                alias, probs = generate_alias_tables(dst_nbs_wt)
            else:
                dst_nbs = (dst_nbs_id, dst_nbs_wt)
                shared_nb_ids = row["shared_neighbor_ids"]
                alias, probs = generate_edge_alias_tables(
                    src, shared_nb_ids, dst_nbs, return_param, inout_param,
                )
            path = extend_random_walk(
                path=row["path"],
                dst_neighbors=dst_nbs_id,
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
