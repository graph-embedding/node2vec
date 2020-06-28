import logging
from typing import Dict
from typing import Any
from typing import Optional
from typing import Tuple
from fugue import FugueWorkflow
from fugue import DataFrame as FugueDataFrame
from fugue import ExecutionEngine as FugueExecutionEngine
from fugue import PandasDataFrame
from fugue_spark import SparkExecutionEngine
from fugue_spark import SparkDataFrame

from node2vec.constants import NODE2VEC_PARAMS
from node2vec.indexer import index_graph_pandas
from node2vec.indexer import index_graph_spark
from node2vec.randomwalk import trim_hotspot_vertices
from node2vec.randomwalk import get_vertex_neighbors
from node2vec.randomwalk import get_edge_shared_neighbors
from node2vec.randomwalk import initiate_random_walk
from node2vec.randomwalk import next_step_random_walk
from node2vec.randomwalk import to_path


#
def trim_index(
    compute_engine: FugueExecutionEngine,
    df_graph: FugueDataFrame,
    indexed: Optional[bool] = False,
    max_out_deg: Optional[int] = 0,
    random_seed: Optional[int] = None,
) -> Tuple[FugueDataFrame, Optional[FugueDataFrame]]:
    """
    The very first steps to treat the input graph:
    1) basic validation of the input graph format: at least have ["src", "dst"] cols,
       it will be an unweighted graph if no "weight" col.
    2) trim some edges to avoid super hotspot vertices: random sampling will be done
       on all the edges of a vertex if the number of edges is greater than a threshold,
       this is critical to reduce data skewness and save disk space
    3) index the graph vertices by using sequential integers to represent vertices,
       this is critical to save memory

    :param compute_engine: an execution engine supported by Fugue
    :param df_graph: the input graph data as general Fugue dataframe
    :param indexed: if the input graph is using sequential integers to note vertices
    :param max_out_deg: the threshold for trimming hotspot vertices, set it to <= 0
                        to turn off trimming
    :param random_seed: optional random seed, for testing only

    Returns a validated, trimmed, and indexed graph
    """
    logging.info("trim_index(): start validating, trimming, and indexing ...")
    if "src" not in df_graph.schema or "dst" not in df_graph.schema:
        raise ValueError(f"Input graph NOT in the right format: {df_graph.schema}")

    params = {"max_out_degree": max_out_deg, "random_seed": random_seed}
    with FugueWorkflow(compute_engine) as dag:
        df = (
            dag.df(df_graph)
            .partition(by=["src"])
            .transform(trim_hotspot_vertices, schema="*", params=params,)
            .compute()
        )
        name_id = None
        if indexed is True:
            return df, name_id
        if isinstance(compute_engine, SparkExecutionEngine):
            df_res, name_id = index_graph_spark(df.native)  # type: ignore
            return SparkDataFrame(df_res), SparkDataFrame(name_id)
        else:
            df_res, name_id = index_graph_pandas(df.as_pandas())
            return PandasDataFrame(df_res), PandasDataFrame(name_id)


#
def random_walk(
    compute_engine: FugueExecutionEngine,
    df_graph: FugueDataFrame,
    n2v_params: Dict[str, Any],
    random_seed: Optional[int] = None,
) -> FugueDataFrame:
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

    :param compute_engine: an execution engine supported by Fugue
    :param df_graph: the input graph data as general Fugue dataframe
    :param n2v_params: the node2vec params
    :param random_seed: optional random seed, for testing only

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
    logging.info("random_walk(): start random walking ...")
    for param in NODE2VEC_PARAMS:
        if param not in n2v_params:
            n2v_params[param] = NODE2VEC_PARAMS[param]
    # create workflow
    df = FugueWorkflow(compute_engine).df(df_graph)
    # process vertices
    df_vertex = df.partition(by=["src"], presort="dst").transform(get_vertex_neighbors)
    df_dst = df_vertex.rename(id="dst", neighbors="dst_neighbors").persist()
    # process edges
    param1 = {"num_walks": n2v_params["num_walks"]}
    df_edge = (
        df[["src", "dst"]]
        .inner_join(df_dst)
        .partition(by=["src"])
        .transform(get_edge_shared_neighbors, params=param1,)
        .persist()
    )
    # conduct random walk with distributed bfs
    walks = df_dst[["dst"]].transform(initiate_random_walk, params=param1).persist()
    param2 = {
        "return_param": n2v_params["return_param"],
        "inout_param": n2v_params["inout_param"],
        "seed": random_seed,
    }
    for _ in range(n2v_params["walk_length"]):
        walks = (
            walks.inner_join(df_dst)
            .inner_join(df_edge)
            .transform(next_step_random_walk, params=param2,)
            .persist()
        )
    # convert paths back to lists
    df_walks = walks.transform(to_path)
    logging.info("random_walk(): random walking done ...")
    return df_walks.compute()
