import logging
from typing import Dict
from typing import Any
from typing import Optional
from fugue import FugueWorkflow
from fugue import DataFrame as FugueDataFrame
from fugue import ExecutionEngine as FugueExecutionEngine

from node2vec.constants import NODE2VEC_PARAMS
from node2vec.randomwalk import calculate_vertex_attributes
from node2vec.randomwalk import calculate_edge_attributes
from node2vec.randomwalk import initiate_random_walk
from node2vec.randomwalk import next_step_random_walk
from node2vec.randomwalk import to_path


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

    with FugueWorkflow(compute_engine) as dag:
        edge_list = dag.df(df_graph).persist()
        # process vertices
        df_vertex = (
            edge_list.partition(by=["src"], presort="dst")
            .transform(calculate_vertex_attributes)
            .persist()
        )

        # process edges
        src = df_vertex[["id", "neighbors"]].rename(id="src", neighbors="src_neighbors")
        dst = df_vertex[["id", "neighbors"]].rename(id="dst", neighbors="dst_neighbors")
        df_edge = edge_list.inner_join(src).inner_join(dst)
        params = n2v_params.copy()
        params.pop("walk_length")
        df_edge = (
            df_edge.partition(by=["src"])
            .transform(calculate_edge_attributes, params=params,)
            .persist()
        )

        # the initial state of random walk
        walks = df_vertex.transform(
            initiate_random_walk, params=dict(num_walks=n2v_params["num_walks"]),
        ).persist()
        for _ in range(n2v_params["walk_length"]):
            walks = (
                walks.inner_join(df_edge)
                .transform(next_step_random_walk, params=dict(seed=random_seed),)
                .persist()
            )

        df_walks = walks.transform(to_path)
        # df_walks.show(12, show_count=True)
        logging.info("random_walk(): random walking done ...")
        return df_walks.compute()
