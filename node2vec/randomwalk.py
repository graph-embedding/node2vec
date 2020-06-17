import logging
import pandas as pd
import networkx as nx
from typing import Iterable
from typing import Dict
from typing import Any
from typing import Optional
from typing import Union
from fugue import FugueWorkflow
from node2vec.constants import NODE2VEC_PARAMS
from node2vec.utils import Neighbors
from node2vec.utils import AliasProb
from node2vec.utils import RandomPath
from node2vec.utils import generate_alias_tables
from node2vec.utils import generate_edge_alias_tables


#
def calculate_vertex_attributes(df: pd.DataFrame,) -> Iterable[Dict[str, Any]]:
    """
    A func to aggregate all neighbors and their weights for every node in the graph

    :param df: a pandas dataframe from a partition of the node dataframe

    return a Iterable of dict, each of which is a row of the result df.
    """
    df = df.sort_values(by=["dst"])
    src = df.loc[0, "src"]
    nbs = Neighbors(df)
    alias_prob = AliasProb(generate_alias_tables(df["weight"].tolist()))
    yield dict(
        id=src, neighbors=nbs.serialize(), alias_prob=alias_prob.serialize(),
    )


#
def calculate_edge_attributes(
    df: pd.DataFrame, num_walks: int, return_param: float, inout_param: float,
) -> Iterable[Dict[str, Any]]:
    """
    A func for running mapPartitions to initiate attributes for every edge in the graph

    :param df:
    :param num_walks:
    :param return_param: defined above
    :param inout_param: defined above
    """
    src = df.loc[0, "src"]
    src_neighbors = df.loc[0, "src_neighbors"]
    alias_obj = AliasProb(generate_alias_tables(src_neighbors))
    for i in range(1, num_walks + 1):
        yield dict(
            src=-i,
            dst=src,
            dst_neighbors=src_neighbors,
            alias_prob=alias_obj.serialize(),
        )
    for row in df:
        dst_neighbors = row["dst_neighbors"]
        alias_prob = generate_edge_alias_tables(
            src, src_neighbors, dst_neighbors, return_param, inout_param
        )
        yield dict(
            src=src,
            dst=row["dst"],
            dst_neighbors=dst_neighbors,
            alias_prob=AliasProb(alias_prob).serialize(),
        )


#
def initiate_random_walk(df: pd.DataFrame, num_walks: int,) -> Iterable[Dict[str, Any]]:
    """
    A func for running mapPartitions to initiate attributes for every node in the graph

    :param df: a pandas dataframe from a partition of the node dataframe
    :param num_walks: int, the number of random walks starting from each vertex

    return a Iterable of dict, each of which is a row of the result df.
    """
    for arow in df:
        src = arow["src"]
        row = {"dst": src}
        for i in range(1, num_walks + 1):
            row.update({"src": -i, "path": RandomPath([-i, src])})
            yield row


#
def next_step_random_walk(
    df: Iterable[Dict[str, Any]],
    nbs_col: str = "dst_neighbors",
    seed: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """

    :param df: the partition of the vertex (random path) dataframe
    :param nbs_col: the name of the dst neighbor col
    :param seed: optional random seed, for testing only

    Extend the random walk path by one more step
    """
    for row in df:
        if row[nbs_col] is not None:
            nbs = Neighbors(row[nbs_col])
            alias_prob = AliasProb(row["alias_prob"])
            path = RandomPath(row["path"])

            _p = path.append(nbs, alias_prob, seed)
            row["path"] = _p.serialize()
            row["src"], row["dst"] = _p.last_edge

        yield row


#
def to_path(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """
    convert a random path from a list to a
    """
    for row in df:
        path = RandomPath(row["path"]).data
        yield dict(src=path[0], walk=path)


#
#
def random_walk(
    dag: FugueWorkflow,
    graph: Union[pd.DataFrame, nx.Graph],
    n2v_params: Dict[str, Any],
    random_seed: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
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

    :param dag: a FugueWorkflow
    :param graph: the input graph data
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

    edge_list = dag.create(graph, schema="src:long,dst:long,weight:double").persist()

    # process vertices
    df_vertex = (
        edge_list.partition(by=["src"])
        .transform(
            calculate_vertex_attributes, schema="id:long,neighbors:str,alias_prob:str",
        )
        .persist()
    )
    # df_vertex.show(1, show_count=True)

    # process edges
    src_df = df_vertex[["id", "neighbors"]].rename(id="src", neighbors="src_neighbors")
    dst_df = df_vertex[["id", "neighbors"]].rename(id="dst", neighbors="dst_neighbors")
    df_edge = edge_list.join(src_df, on=["src"], how="inner").join(
        dst_df, on=["dst"], how="inner"
    )
    df_edge = (
        df_edge.partition(by=["src"])
        .transform(
            calculate_edge_attributes,
            schema="src:long,dst:long,dst_neighbors:str,alias_prob:str",
            params=dict(
                num_walks=n2v_params["num_walks"],
                return_param=n2v_params["return_param"],
                inout_param=n2v_params["inout_param"],
            ),
        )
        .persist()
    )
    # df_edge.show(1,show_count=True)

    # the initial state of random walk
    walks = df_vertex.transform(
        initiate_random_walk,
        schema="src:long,dst:long,path:str",
        params=dict(num_walks=n2v_params["num_walks"]),
    ).persist()
    for _ in range(n2v_params["walk_length"]):
        walks = walks.join(df_edge, on=["src", "dst"], how="inner").transform(
            next_step_random_walk, schema="*", params=dict(seed=random_seed),
        )
        walks = walks[["src", "dst", "path"]].persist()
        # walks.show(1)
    df_walks = walks.transform(to_path, schema="*")
    df_walks.show(show_count=True)
    logging.info("random_walk(): random walking done ...")
    return df_walks
