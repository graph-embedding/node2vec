#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from node2vec import __version__


with open("README.md") as readme_file:
    LONG_DESCRIPTION = readme_file.read()

setup(
    name="node2vec-fugue",
    version=__version__,
    description="Highly Scalable Distributed Node2Vec Algorithm Library",
    long_description=LONG_DESCRIPTION,
    url="http://github.com/fugue-project/node2vec",
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Jintao Zhang",
    author_email="jtzhang@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    keywords="graph embedding",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3 :: Only",
    ],
    install_requires=["PyYAML", "future", "pathlib2", "six", "joblib"],
)
