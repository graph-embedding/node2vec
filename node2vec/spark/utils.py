import random
from pyspark.sql import Row
from typing import List
from typing import Tuple
from typing import Dict
from typing import Callable


def generate_alias_tables(
    node_weights: List[Tuple[int, float]],
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
    avg_weight = sum(x for _, x in node_weights) / n
    probs = [float(weight) / avg_weight for _, weight in node_weights]
    alias = [0 for _ in range(n)]
    underfull = []
    overfull = []
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


def generate_alias_tables_wiki(
    node_weights: List[Tuple[int, float]],
) -> Tuple[List[int], List[float]]:
    """
    Generate the two utility table for the Alias Method, an efficient algorithm for
    sampling from a non-uniform discrete probability distribution. Refer to the wiki
    page https://en.wikipedia.org/wiki/Alias_method for detailed algorithm.

    :param node_weights: a list of neighboring nodes and their weights

    return the two utility tables as lists
        probs: the probability table holding the relative probability of each neighbor
        alias: the alias table holding the alias index to be sampled from
    """
    n = len(node_weights)
    avg_weight = sum(x for _, x in node_weights) / n
    probs = [float(weight) / avg_weight for _, weight in node_weights]
    alias = [-1 for _ in range(n)]
    underfull = []
    overfull = []
    for i in range(n):
        if abs(probs[i] - 1.0) <= 1e-7:  # if probs[i] ~ 1.0
            alias[i] = i
        elif probs[i] > 1.0:
            overfull.append(i)
        else:
            underfull.append(i)

    while underfull and overfull:
        under, over = underfull.pop(), overfull.pop()
        alias[under] = over
        probs[over] = probs[over] + probs[under] - 1.0
        if abs(probs[over] - 1.0) <= 1e-7:  # if probs[over] ~ 1.0
            if alias[over] < 0:
                alias[over] = over
        elif probs[over] > 1.0:
            overfull.append(over)
        else:
            underfull.append(over)
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
    neighbors_dst: List[Tuple[int, float]] = []
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
        neighbors_dst.append((dst_neighbor_id, unnorm_prob))
    return generate_alias_tables(neighbors_dst)


def sampling_from_alias_original(alias: List[int], probs: List[float],) -> int:
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.

    keep aligned with the original author's implementation to help parity tests.

    :param alias: the alias list in range [0, n)
    :param probs: the pseudo-probability table

    return the picked index in the neighbor list as next node in the random walk path.
    """
    pick = int(random.random() * len(alias))
    if random.random() < probs[pick]:
        return pick
    else:
        return alias[pick]


def sampling_from_alias(randval: float, alias: List[int], probs: List[float],) -> int:
    """
    Draw sample from a non-uniform discrete distribution using alias sampling. This
    Alias method implementation is more aligned with the wiki description.

    :param randval: a uniform random float number in [0.0, 1.0], be careful on the
                    right ending point as a special handling might be required.
    :param alias: the alias list in range [0, n)
    :param probs: the pseudo-probability table

    return the picked index in the neighbor list as next node in the random walk path.
    """
    if randval == 1.0:  # handle special case when random val == 1.0
        randval -= 1e-7
    n = len(alias)
    pick = int(n * randval)
    y = n * randval - pick
    if y < probs[pick]:
        return pick
    else:
        return alias[pick]


def next_step_random_walk(
    path: List[int],
    randval: float,
    dst_neighbors: List[int],
    alias: List[int],
    probs: List[float],
) -> List[int]:
    """
    Extend the random walk path by making a biased random sampling at next step

    :param path: the existing random walk path
    :param randval: a uniform random float number in [0.0, 1.0], be careful on the
                    right ending point as a special handling might be required.
    :param dst_neighbors: the neighbor node id's of the dst node
    :param alias: the pre-calculated alias list
    :param probs: the pre-calculated probs list

    return the extended path with one more node in the random walk path.
    """
    next_index = sampling_from_alias(randval, alias, probs)
    next_vertex = dst_neighbors[next_index]
    # first step
    if len(path) == 2 and path[0] < 0:
        path = [path[1], next_vertex]
    # remaining step
    else:
        path.append(next_vertex)
    return path


def aggregate_vertex_neighbors(partition: List[Row]) -> List[Row]:
    """
    A func to aggregate all neighbors and their weights for every node in the graph

    :param partition: a partition in List[Row] of the node dataframe
    return a List[Row] after calculating attributes as a partition ["id", "neighbors"]
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
        result += [Row(id=src, neighbors=neighbors)]
    return result


def calculate_vertex_attributes(num_walks: int) -> Callable:
    """
    Creates a func for running mapPartitions to initiate attributes for every edge
    in the graph

    :param num_walks: the number of walks starting from each vertex
    """

    def _get_node_attr_each_partition(partition: List[Row]) -> List[Row]:
        """
        The inner func to process vertices in each partition

        :param partition: a partition in List[Row] of the node dataframe

        The resulting dataframe has two node attributes:
            neighbors: List[Tuple[int, float]] = []
            path: List[int] = []
        return a List[Row] with calculated attributes as a partition of the result df.
        """
        src_neighbors: Dict[int, List[Tuple[int, float]]] = {}
        id_name: Dict[int, str] = {}
        for arow in partition:
            row = arow.asDict()
            src = row["src"]
            if src not in src_neighbors:
                src_neighbors[src] = []
            src_neighbors[src].append((row["dst"], row["weight"]))
            if "name" in row:
                id_name[src] = row["name"]

        result: List[Row] = []
        for src, neighbors in src_neighbors.items():
            neighbors.sort(key=lambda x: x[0])
            dic = {"id": src, "neighbors": neighbors}
            if src in id_name:
                dic.update({"name": id_name[src]})
            for i in range(1, num_walks + 1):
                dic.update({"path": [-i, src]})
                result += [Row(**dic)]
        return result

    return _get_node_attr_each_partition


def calculate_edge_attributes(
    return_param: float, inout_param: float, num_walks: int,
) -> Callable:
    """
    A func for running mapPartitions to initiate attributes for every edge in the graph

    :param return_param: defined above
    :param inout_param: defined above
    :param num_walks: int
    """

    def _get_edge_attr_each_partition(partition: List[Row]) -> List[Row]:
        """
        the inner wrapper func to process edges in each partition

        :param partition: a partition in List[Row] of the node dataframe

        The resulting dataframe has three edge attributes:
             dst_neighbors: List[int] = []
             alias: List[int] = []
             probs: List[float] = []
        return a List[Row] after calculating attributes as a partition of the result.
        """
        result: List[Row] = []
        src_to_neighbors: Dict[int, List[Tuple[int, float]]] = {}
        for arow in partition:
            row = arow.asDict()
            src, dst_neighbors = row["src"], row["dst_neighbors"]
            src_to_neighbors[src] = row["src_neighbors"]
            alias, probs = generate_edge_alias_tables(
                src, row["src_neighbors"], dst_neighbors, return_param, inout_param
            )
            result += [
                Row(
                    edge=str(src) + " " + str(row["dst"]),
                    dst_neighbors=[v[0] for v in dst_neighbors],
                    alias=alias,
                    probs=probs,
                )
            ]

        for src, neighbors in src_to_neighbors.items():
            alias, probs = generate_alias_tables(neighbors)
            for i in range(1, num_walks + 1):
                result += [
                    Row(
                        edge=str(-i) + " " + str(src),
                        dst_neighbors=[v[0] for v in neighbors],
                        alias=alias,
                        probs=probs,
                    )
                ]
        return result

    return _get_edge_attr_each_partition
