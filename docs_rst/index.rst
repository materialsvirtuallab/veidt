Veidt - A materials science deep learning library
=================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Veidt is a deep learning library for materials science. It builds on top of
the popular pymatgen (Python Materials Genomics) materials analysis library
and well-known deep learning libraries like Keras and Tensorflow. The aim is
to link the power of both kinds of libraries for rapid experimentation and
learning of materials data.

Installation
============

Veidt should be easily installable via pip on most systems::

    pip install veidt

Changes
=======

v0.0.2
------
* ELSIE spectrum matching algorithm.
* Improvements to NeuralNetwork and new LinearModel.

v0.0.1
------
* Initial release of basic abstract models and implementation of neural net.

General concepts
================

Veidt works by abstracting some common tasks to high-level classes. For example,
deep learning require numerical representations, i.e., descriptors. Veidt
specifies Describer classes that take an input object and convert it into a
numerical representation. Similarly, the Model classes serves as a wrapper
around common models.

Here's a simple example utilizing Materials Project::

    from pymatgen import MPRester
    from veidt.descriptors import DistinctSiteProperty
    from veidt.models import NeuralNet

    # Let's grab the Li2O and Na2O structures via pymatgen's high level
    # interface to the Materials Project API.
    mpr = MPRester()
    li2o = mpr.get_structures("Li2O")[0]
    na2o = li2o.copy()
    na2o.replace_species({"Li": "Na"})

    # Construct a NeuralNet with a single hidden layer of 20 neurons.
    # The DistinctSiteProperty just says we want the look at only the 8c sites
    # and use the atomic number (Z) of the site as a descriptor. This is not
    # a good model of course. It is meant to illustrate the concepts.
    model = NeuralNet([20], describer=DistinctSiteProperty(['8c'], ["Z"]))

    # Create some artificial data to fit.
    structures = [li2o] * 100 + [na2o] * 100
    energies = [3] * 100 + [4] * 100

    # Fit the model.
    model.fit(inputs=structures, outputs=energies, epochs=100)

    # Use the model to do a prediction.
    model.predict([na2o])

API docs
========

.. toctree::
   :maxdepth: 4

   veidt

License
=======

Veidt is released under the BSD License::

    Copyright (c) 2015, Regents of the University of California
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
