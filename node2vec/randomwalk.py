import logging
import random
import pickle
import base64
import pandas as pd
import networkx as nx
from typing import Iterable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Any
from typing import Optional
from typing import Union
from fugue.dag.workflow import FugueWorkflow
from node2vec.constants import NODE2VEC_PARAMS


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


def generate_edge_alias_tables(
    src_id: int,
    src_neighbors: List[Tuple[int, float]],
    dst_neighbors: List[Tuple[int, float]],
    return_param: float = 1.0,
    inout_param: float = 1.0,
) -> Tuple[List[int], List[float]]:
    """
    Apply the biased sampling on edge weight described in the node2vec paper. Each entry
    here represents an edge, and the src and dst node's info.

    :param src_id: the source node id
    :param src_neighbors: the list of source node's neighbor node id's and weights
    :param dst_neighbors: the list of destination node's neighbor node id's and weights
    :param return_param: the parameter p defined in the paper
    :param inout_param: the parameter q defined in the paper

    return the utility tables of the Alias method, after weights are biased.
    """
    if return_param == 0 or inout_param == 0:
        raise ValueError(
            f"Zero return ({return_param}) or inout ({inout_param}) parameter!"
        )
    src_neighbor_ids = {sid for sid, _ in src_neighbors}
    # apply bias to edge weights
    neighbors_dst: List[float] = []
    for dst_neighbor_id, weight in dst_neighbors:
        # go back to the src id
        if dst_neighbor_id == src_id:
            unnorm_prob = weight / return_param
        # go to a neighbor of src
        elif dst_neighbor_id in src_neighbor_ids:
            unnorm_prob = weight
        # go to a brand new vertex
        else:
            unnorm_prob = weight / inout_param
        neighbors_dst.append(unnorm_prob)
    return generate_alias_tables(neighbors_dst)


class Neighbors(object):
    def __init__(self, obj):
        if isinstance(obj, str):
            self.data = pickle.loads(base64.b64decode(obj.encode()))
        elif isinstance(obj, pd.DataFrame):
            self.data = (obj["dst"].tolist(), obj["weight"].tolist())
        elif isinstance(obj, tuple):
            self.data = obj
        self._keys = None

    @property
    def keys(self):
        if self._keys is None:
            self._keys = set(self.data[0])
        return self._keys

    def __contains__(self, id):
        return id in self.keys

    def items(self):
        return zip(self.data[0], self.data[1])

    def serialize(self):
        return base64.b64encode(pickle.dumps(self.data)).decode()

    def as_pandas(self):
        return pd.DataFrame({"dst": self.data[0], "weight": self.data[1]})


class AliasProb(object):
    def __init__(self, obj):
        if isinstance(obj, str):
            self._data = pickle.loads(base64.b64decode(obj.encode()))
        elif isinstance(obj, pd.DataFrame):
            self._data = (obj["alias"].tolist(), obj["probs"].tolist())
        elif isinstance(obj, tuple):
            self._data = obj

    @property
    def alias(self):
        """
        alias: the alias list in range [0, n)
        """
        return self._data[0]

    @property
    def probs(self):
        """
        probs: the pseudo-probability table
        """
        return self._data[1]

    def serialize(self):
        return base64.b64encode(pickle.dumps(self._data)).decode()

    def draw_alias(
            self,
            nbs: Neighbors,
            seed: Optional[int] = None,
    ) -> int:
        """
        Draw sample from a non-uniform discrete distribution using alias sampling.
        Keep aligned with the original author's implementation to help parity tests.

        :param nbs: a Neighbors object
        :param seed: a int as the random sampling seed, for testing only.
        Return the picked index in the neighbor list as next node of the random path.
        """
        if seed is not None:
            random.seed(seed)
        pick = int(random.random() * len(self.alias))
        if random.random() < self.probs[pick]:
            return nbs.data[0][pick]
        else:
            return nbs.data[0][self.alias[pick]]


class RandomPath(object):
    def __init__(self, obj):
        if isinstance(obj, str):
            self.data = pickle.loads(base64.b64decode(obj.encode()))
        elif isinstance(obj, list):
            self.data = obj

    @property
    def path(self):
        return self.data

    @property
    def last_edge(self):
        return self.data[-2], self.data[-1]

    def append(
            self,
            dst_neighbors: Neighbors,
            alias_prob: AliasProb,
            seed: Optional[int] = None,
    ) -> "RandomPath":
        """
        Extend the random walk path by making a biased random sampling at next step

        :param dst_neighbors: the neighbor node id's of the dst node
        :param alias_prob:
        :param seed: optional random seed, for testing only
        Return the extended path with one more node in the random walk path.
        """
        next_vertex = alias_prob.draw_alias(dst_neighbors, seed)
        path = list(self.data)
        # first step
        if len(path) == 2 and path[0] < 0:
            path = [path[1], next_vertex]
        # remaining step
        else:
            path.append(next_vertex)
        yield RandomPath(path)

    def serialize(self):
        return base64.b64encode(pickle.dumps(self.data)).decode()

    def __str__(self):
        return self.data.__repr__()


#
def calculate_vertex_attributes(
        df: pd.DataFrame,
) -> Iterable[Dict[str, Any]]:
    """
    A func to aggregate all neighbors and their weights for every node in the graph

    :param df: a pandas dataframe from a partition of the node dataframe

    return a Iterable of dict, each of which is a row of the result df.
    """
    df = df.sort_values(by=['dst'])
    src = df.loc[0, "src"]
    nbs = Neighbors(df)
    alias_prob = AliasProb(
        generate_alias_tables(df["weight"].tolist())
    )
    yield dict(id=src,
               neighbors=nbs.serialize(),
               alias_prob=alias_prob.serialize(),
               )


#
def calculate_edge_attributes(
        df: pd.DataFrame,
        num_walks: int,
        return_param: float,
        inout_param: float,
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
        yield dict(src=-i,
                   dst=src,
                   dst_neighbors=src_neighbors,
                   alias_prob=alias_obj.serialize(),
                   )
    for row in df:
        dst_neighbors = row['dst_neighbors']
        alias_prob = generate_edge_alias_tables(
            src, src_neighbors, dst_neighbors, return_param, inout_param
        )
        yield dict(src=src,
                   dst=row["dst"],
                   dst_neighbors=dst_neighbors,
                   alias_prob=AliasProb(alias_prob).serialize(),
                   )


#
def initiate_random_walk(
        df: pd.DataFrame,
        num_walks: int,
) -> Iterable[Dict[str, Any]]:
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
    df_vertex = edge_list.partition(by=["src"]).transform(
        calculate_vertex_attributes, schema="id:long,neighbors:str,alias_prob:str",
    ).persist()
    # df_vertex.show(1, show_count=True)

    # process edges
    src_df = df_vertex[["id", "neighbors"]].rename(id="src", neighbors="src_neighbors")
    dst_df = df_vertex[["id", "neighbors"]].rename(id="dst", neighbors="dst_neighbors")
    df_edge = (edge_list.join(src_df, keys=["src"], how="inner")
                        .join(dst_df, keys=["dst"], how="inner")
               )
    df_edge = df_edge.partition(by=["src"]).transform(
        calculate_edge_attributes,
        schema="src:long,dst:long,dst_neighbors:str,alias_prob:str",
        params=dict(num_walks=n2v_params["num_walks"],
                    return_param=n2v_params["return_param"],
                    inout_param=n2v_params["inout_param"]),
    ).persist()
    # df_edge.show(1,show_count=True)

    # the initial state of random walk
    walks = df_vertex.transform(
        initiate_random_walk,
        schema="src:long,dst:long,path:str",
        params=dict(num_walks=n2v_params["num_walks"]),
    ).persist()
    for _ in range(n2v_params["walk_length"]):
        walks = walks.join(df_edge, keys=["src", "dst"], how="inner").transform(
            next_step_random_walk, schema="*", params=dict(seed=random_seed),
        )
        walks = walks[["src", "dst", "path"]].persist()
        # walks.show(1)
    df_walks = walks.transform(to_path, schema="*").show(show_count=True)
    df_walks.show(1)
    logging.info("random_walk(): random walking done ...")
    return df_walks
