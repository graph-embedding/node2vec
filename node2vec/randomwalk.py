import random
import pickle
import base64
import pandas as pd
from typing import List
from typing import Tuple
from typing import Union
from typing import Iterable
from typing import Dict
from typing import Any
from typing import Optional

from node2vec.constants import MAX_OUT_DEGREES


class Neighbors(object):
    def __init__(self, obj: Union[str, pd.DataFrame, Tuple[List[int], List[float]]]):
        if isinstance(obj, str):
            self._data = pickle.loads(base64.b64decode(obj.encode()))
        elif isinstance(obj, pd.DataFrame):
            self._data = (obj["dst"].tolist(), obj["weight"].tolist())
        else:
            self._data = obj

    @property
    def dst_id(self):
        return self._data[0]

    @property
    def dst_wt(self):
        return self._data[1]

    def items(self):
        return zip(self._data[0], self._data[1])

    def serialize(self):
        return base64.b64encode(pickle.dumps(self._data)).decode()

    def as_pandas(self):
        return pd.DataFrame({"dst": self._data[0], "weight": self._data[1]})


class AliasProb(object):
    def __init__(self, obj: Union[str, pd.DataFrame, Tuple[List[int], List[float]]]):
        if isinstance(obj, str):
            self._data = pickle.loads(base64.b64decode(obj.encode()))
        elif isinstance(obj, pd.DataFrame):
            self._data = (obj["alias"].tolist(), obj["probs"].tolist())
        else:
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

    def draw_alias(self, nbs: Neighbors, seed: Optional[int] = None,) -> int:
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
            return nbs.dst_id[pick]
        else:
            return nbs.dst_id[self.alias[pick]]


class RandomPath(object):
    def __init__(self, obj: Union[str, List[int]]):
        if isinstance(obj, str):
            self._data = pickle.loads(base64.b64decode(obj.encode()))
        else:
            self._data = obj

    @property
    def path(self):
        return self._data

    @property
    def last_edge(self):
        return self._data[-2], self._data[-1]

    def serialize(self):
        return base64.b64encode(pickle.dumps(self._data)).decode()

    def __str__(self):
        return self._data.__repr__()

    def append(
        self,
        dst_neighbors: Neighbors,
        alias_prob: AliasProb,
        seed: Optional[int] = None,
    ):
        """
        Extend the random walk path by making a biased random sampling at next step

        :param dst_neighbors: the neighbor node id's of the dst node
        :param alias_prob:
        :param seed: optional random seed, for testing only
        Return the extended path with one more node in the random walk path.
        """
        next_vertex = alias_prob.draw_alias(dst_neighbors, seed)
        path = list(self._data)
        # first step
        if len(path) == 2 and path[0] < 0:
            path = [path[1], next_vertex]
        # remaining step
        else:
            path.append(next_vertex)
        return RandomPath(path)


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
# ============================- transformer func ====================================
#
def trim_hotspot_vertices(
    df: pd.DataFrame, max_out_degree: int = 0, random_seed: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """
    This func is to do random sampling on the edges of vertices which have very large
    number of out edges. A maximal threshold is provided and random sampling is applied.
    By default the threshold is 100,000.

    :param df: a pandas dataframe from a partition of the node dataframe
    :param max_out_degree: the max out degree of each vertex to avoid hotspot
    :param random_seed: the seed for random sampling, testing only

    return a Iterable of dict, each of which is a row of the result df.
    """
    if max_out_degree <= 0:
        max_out_degree = MAX_OUT_DEGREES
    if len(df["dst"].tolist()) > max_out_degree:
        if random_seed is not None:
            df = df.sample(n=max_out_degree, random_state=random_seed)
        else:
            df = df.sample(n=max_out_degree)
    for _, row in df.iterrows():
        yield dict(row)


# schema: id:int,neighbors:str
def get_vertex_neighbors(df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    """
    Aggregate all neighbors and their weights for every vertex in the graph

    :param df: a pandas dataframe from a partition of the node dataframe
    return a Iterable of dict, each of which is a row of the result df.
    """
    src = df.loc[0, "src"]
    nbs = Neighbors(df)
    yield dict(id=src, neighbors=nbs.serialize())


# schema: src:int,dst:int,shared_neighbor_ids:[int]
def get_edge_shared_neighbors(
    df: pd.DataFrame, num_walks: int,
) -> Iterable[Dict[str, Any]]:
    """
    Get the shared neighbors of the src and dst vertex of every edge in the graph

    :param df: a pandas df with cols "src", "dst", "dst_neighbors"
    :param num_walks: the num of random walks starting from each vertex
    """
    src = df.loc[0, "src"]
    for i in range(1, num_walks + 1):
        yield dict(src=-i, dst=src, shared_neighbor_ids=[])

    src_neighbors = set(df["dst"].tolist())
    for _, row in df.iterrows():
        dst_nbs_id = Neighbors(row["dst_neighbors"]).dst_id
        shared_ids = [x for x in dst_nbs_id if x in src_neighbors]
        yield dict(src=row["src"], dst=row["dst"], shared_neighbor_ids=shared_ids)


# schema: src:int,dst:int,path:[int]
def initiate_random_walk(
    df: Iterable[Dict[str, Any]], num_walks: int,
) -> Iterable[Dict[str, Any]]:
    """
    Initiate the random walk path and replicate for num_walks times

    :param df: a pandas dataframe from a partition of the node dataframe
    :param num_walks: int, the number of random walks starting from each vertex

    return a Iterable of dict, each of which is a row of the result df.
    """
    for arow in df:
        src = arow["dst"]
        row = {"dst": src}
        for i in range(1, num_walks + 1):
            row.update({"src": -i, "path": [-i, src]})
            yield row


# schema: src:int,dst:int,path:[int]
def next_step_random_walk(
    df: Iterable[Dict[str, Any]],
    return_param: float,
    inout_param: float,
    seed: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Extend the random walk path by one more step

    :param df: the partition of the vertex (random path) dataframe
    :param return_param:
    :param inout_param:
    :param seed: optional random seed, for testing only
    """
    for row in df:
        src = row["src"]
        dst_nbs = Neighbors(row["dst_neighbors"])
        if src < 0:
            alias_prob = AliasProb(generate_alias_tables(dst_nbs.dst_wt))
        else:
            dst_nbs_data = (dst_nbs.dst_id, dst_nbs.dst_wt)
            shared_nb_ids = row["shared_neighbor_ids"]
            alias_prob = AliasProb(
                generate_edge_alias_tables(
                    src, shared_nb_ids, dst_nbs_data, return_param, inout_param,
                )
            )
        path = RandomPath(row["path"])
        _p = path.append(dst_nbs, alias_prob, seed)
        yield dict(src=_p.last_edge[0], dst=_p.last_edge[1], path=_p.path)


# schema: src:int,walk:[int]
def to_path(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """
    convert a random path from a list to a pair [src, walk]
    """
    for row in df:
        path = RandomPath(row["path"]).path
        yield dict(src=path[0], walk=path)
