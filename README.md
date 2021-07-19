# Node2Vec

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/node2vec-fugue.svg)](https://pypi.python.org/pypi/node2vec-fugue/)
[![PyPI license](https://img.shields.io/pypi/l/node2vec-fugue.svg)](https://pypi.python.org/pypi/node2vec-fugue/)
[![PyPI version](https://badge.fury.io/py/node2vec-fugue.svg)](https://pypi.python.org/pypi/node2vec-fugue/)
[![Coverage Status](https://coveralls.io/repos/github/fugue-project/node2vec/badge.svg?branch=master)](https://coveralls.io/github/fugue-project/node2vec?branch=master)

A highly scalable distributed node2vec algorithm

## Installation
```
pip install node2vec-fugue
```


## Release History

### 0.3.5
* Fix bugs on checkpointing paths of deep traversal

### 0.3.4
* support checkpointing using fugue for deep traversal
* add a node2vec implementation in native spark
* add two working examples in fugue spark and native spark

### 0.3.1
* 1st open-source version
* highly scalable to graph with hundreds of millions of vertices and billions of edges
* Can handle highly sparse graphs and skewed graphs

### 0.2.13
* Refactor and add native PySpark node2vec

### 0.2.9
* alternative persist in bfs
* improve alias calculation

### 0.2.8
* Significant improvement on handling hotspot vertices
* Fix misuse of Fugue compute()

### 0.2.5
* Add indexer of graph vertices
* Allow trimming hotspot vertices

### 0.2.4
* Use Apache-2.0 license

### 0.2.3
* Add graph indexer for arbitrary vertex names
* Refactor layout

### 0.2.2
* support word2vec on either gensim or spark.ml.feature backend
* fully tested

### 0.2.1
* change the interface to support backend compute engine
* use iterable to replace most pandas usage

### 0.2.0
* support fugue based node2vec
* not yet support input format validation and vertex indexing
