GRANDlib
========

Offline data handling, simulation and analysis for the `Giant Radio Array for
Neutrino Detection <http://grand.cnrs.fr>`_.

GRANDlib performs end-to-end simulation of the detector: from an electric
field produced by an external air-shower simulation, through the antenna
response, the Galactic noise and the radio-frequency chain, to the digitized
voltage a detection unit records.  It also defines the data format the
collaboration uses, and the coordinate systems everything is expressed in.

.. note::

   **This documentation is under construction.**  It is being rebuilt as part
   of the repository overhaul: a single Sphinx tree replacing five scattered
   sources, an API reference generated from the docstrings, and narrative
   pages whose examples are executed when the page is built.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   coordinates
   datamodel
   known_issues

.. toctree::
   :maxdepth: 2
   :caption: Reference

   architecture
   api

Citing
------

R. Alves Batista *et al.* (the GRAND Collaboration), *GRANDlib: A simulation
pipeline for the Giant Radio Array for Neutrino Detection (GRAND)*,
`arXiv:2408.10926 <https://arxiv.org/abs/2408.10926>`_.
