import random
import pickle
import base64
import pandas as pd
from typing import List
from typing import Tuple
from typing import Optional


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

    def __contains__(self, _id):
        return _id in self.keys

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
    ):
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
