from typing import Dict
from typing import Any


# num of partitions for map partitions
NUM_PARTITIONS: int = 2000


# the default word2vec parameters, except the vectorSize to be set explicitly
WORD2VEC_PARAMS: Dict[str, Any] = {
    # min token freq to be included in the word2vec model's vocabulary
    "minCount": 10,
    # num of partitions for sentences of words
    "numPartitions": 100,
    "stepSize": 0.025,
    "maxIter": 10,
    "seed": None,
    # num of neighborhood nodes (context [-window, window])
    "windowSize": 5,
    # max length (in words) of each sentence in the input data. Any sentence longer
    # than this threshold will be divided into chunks up to the size
    "maxSentenceLength": 10000,
}
