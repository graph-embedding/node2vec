import random
from pyspark.sql import Row
from typing import List
from typing import Tuple
from typing import Dict
from typing import Set
from typing import Optional
from typing import Callable


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
