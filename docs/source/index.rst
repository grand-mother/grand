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
   simulation
   data_files
   sim2root
   troubleshooting
   notebooks

.. toctree::
   :maxdepth: 2
   :caption: Reference

   architecture
   implementation
   api

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   roadmap
   ci
   testing
   known_issues

.. toctree::
   :maxdepth: 2
   :caption: Handbook

   handbook/index

.. tip::

   The GRANDlib Handbook is included in full under
   :doc:`Handbook <handbook/index>`, and is also available as a
   :download:`PDF (78 pages) <_static/GRANDlib_Handbook.pdf>`.  Both are built
   from the same LaTeX source and carry the same errata; the PDF's title page
   records the commit it was compiled from.

.. toctree::
   :maxdepth: 1
   :caption: Project

   glossary
   references
   citing
   changelog

Citing
------

R. Alves Batista *et al.* (the GRAND Collaboration), *GRANDlib: A simulation
pipeline for the Giant Radio Array for Neutrino Detection (GRAND)*,
`arXiv:2408.10926 <https://arxiv.org/abs/2408.10926>`_.
