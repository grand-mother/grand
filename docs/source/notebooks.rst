Tutorial notebooks
==================

Worked notebooks live in `notebooks/
<https://github.com/grand-mother/grand/tree/dev-next/notebooks>`_, numbered in
reading order.  Each carries its figures inline, so they can be read on GitHub
without being run.

They are the long form of the narrative pages.  A page states a convention and
shows it executing in a few lines; a notebook works through the same material
with the reasoning around it — why the convention is what it is, what happens
at the edges, and what the numbers were checked against.

To run them rather than read them::

    conda env create -f env/conda/grand-dev.yml --solver=libmamba
    conda activate grand-dev
    source env/setup.sh
    jupyter lab notebooks/

.. note::

   The notebooks are **not** built into this documentation.  Executing them on
   every documentation build would be slow, and their stored outputs are more
   useful than freshly computed ones for reading.  A scheduled job runs them
   weekly instead, so they cannot rot unnoticed.

Available
---------

`01. Coordinate systems <https://github.com/grand-mother/grand/blob/dev-next/notebooks/01_coordinates.ipynb>`_
   The four frames and the conversions between them; the two conventions that
   most often catch people out; a detector layout drawn in ``GRANDCS`` and
   again in geodetic coordinates; a shower axis in both.  The long form of
   :doc:`coordinates`.
