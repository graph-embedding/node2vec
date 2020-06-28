from typing import Dict
from typing import Any


# num of partitions for map partitions
NUM_PARTITIONS: int = 3000


# the default node2vec parameters
NODE2VEC_PARAMS: Dict[str, Any] = {
    #  num of walks starting from each node on sampling random walks, [10, 30]
    "num_walks": 10,
    #  length of each random walk path, [10, 30]
    "walk_length": 20,
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
    "minCount": 10,
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
