def test_node2vec_params() -> None:
    """
    test dict constant NODE2VEC_PARAMS
    """
    from node2vec.constants import NODE2VEC_PARAMS

    assert isinstance(NODE2VEC_PARAMS, dict)
    for param in ["num_walks", "walk_length"]:
        assert param in NODE2VEC_PARAMS and isinstance(NODE2VEC_PARAMS[param], int)
    for param in ["return_param", "inout_param"]:
        assert param in NODE2VEC_PARAMS and isinstance(NODE2VEC_PARAMS[param], float)


def test_word2vec_params() -> None:
    """
    test dict constant NODE2VEC_PARAMS
    """
    from node2vec.constants import WORD2VEC_PARAMS

    assert isinstance(WORD2VEC_PARAMS, dict)
    assert "stepSize" in WORD2VEC_PARAMS
    assert isinstance(WORD2VEC_PARAMS["stepSize"], float)

    for param in ["minCount", "numPartitions", "maxIter", "maxSentenceLength",
                  "windowSize", "vectorSize"]:
        assert param in WORD2VEC_PARAMS and isinstance(WORD2VEC_PARAMS[param], int)


def test_gensim_params() -> None:
    """
    test dict constant GENSIM_PARAMS
    """
    from node2vec.constants import GENSIM_PARAMS

    assert isinstance(GENSIM_PARAMS, dict)
    assert "alpha" in GENSIM_PARAMS and isinstance(GENSIM_PARAMS["alpha"], float)

    for param in ["min_count", "iter", "batch_words", "window", "size",
                  "negative", "workers"]:
        assert param in GENSIM_PARAMS and isinstance(GENSIM_PARAMS[param], int)
