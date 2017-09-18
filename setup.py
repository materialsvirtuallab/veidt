# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from setuptools import setup, find_packages

setup(
    name="veidt",
    packages=find_packages(),
    version="0.0.2",
    install_requires=["numpy", "scipy", "monty", "keras", "tensorflow",
                      "scikit-learn", "pymatgen"],
    author="Materials Virtual Lab",
    author_email="ongsp@eng.ucsd.edu",
    maintainer="Shyue Ping Ong",
    maintainer_email="ongsp@eng.ucsd.edu",
    url="http://www.materialsvirtuallab.org",
    license="BSD",
    description="Veidt is a deep learning library for materials science.",
    long_description="""Veidt is a deep learning library for materials science. It builds on top of
the popular pymatgen (Python Materials Genomics) materials analysis library
and well-known deep learning libraries like Keras and Tensorflow. The aim is
to link the power of both kinds of libraries for rapid experimentation and
learning of materials data.""",
    keywords=["materials", "science", "deep", "learning"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
