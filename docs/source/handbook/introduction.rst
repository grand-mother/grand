.. This page is generated from resources/GRANDlib_Handbook.zip
   by docs/dev/build_handbook.py.  Do not edit it by hand.

Introduction
============

.. warning::

   This section contains statements the code contradicts.  See :doc:`index` for the errata.


`GRANDlib <https://arxiv.org/abs/2408.10926>`__ is the official Python-based software suite developed by the GRAND Collaboration (Giant Radio Array for Neutrino Detection). It allows as to simulate air showers, convert electric fields to RF voltages, manage terrain and antenna response, process ROOT files and visualize results. This handbook provides an in-depth guide for practical, project-based use.

The software provides tools to:

#. Simulate air showers and generate corresponding electric field traces.

#. Convert electric fields to voltages using antenna models and RF chains.

#. Process, analyze, and visualize voltage traces and electric fields.

#. Interface with ROOT files containing real or simulated data.

#. Load terrain and detector configurations using topographic data.

GRANDlib is modular, open-source, and supports both centralized and local data analysis workflows. It is compatible with large computing clusters as well as standalone systems using Docker or Conda environments.

Its structure is divided into multiple Python modules, each corresponding to a specific task, including:

- ``grand.io``: file I/O and event access

- ``grand.aoi``: analysis-oriented tools for filtering and reconstruction

- ``grand.topography``: terrain data handling

- ``efield2voltage``: E-field to voltage conversion using antenna responses

- ``RFChain``: modeling the RF electronics response

This handbook offers a comprehensive guide to installing, configuring, and using GRANDlib for both scientific and technical applications. It is targeted at users who want to contribute to the GRAND project or apply its tools in cosmic-ray and neutrino detection workflows.

