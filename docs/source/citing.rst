How to cite
===========

If GRANDlib contributes to work you publish, please cite the GRANDlib paper
:cite:`GRAND:2024atu`.

.. code-block:: bibtex

   @article{GRAND:2024atu,
       author        = "Alves Batista, Rafael and others",
       collaboration = "GRAND",
       title         = "{GRANDlib: A simulation pipeline for the Giant Radio
                        Array for Neutrino Detection (GRAND)}",
       eprint        = "2408.10926",
       archivePrefix = "arXiv",
       primaryClass  = "astro-ph.IM",
       doi           = "10.1016/j.cpc.2024.109461",
       journal       = "Comput. Phys. Commun.",
       volume        = "308",
       pages         = "109461",
       year          = "2025"
   }

The entry above is INSPIRE's, so its key and formatting match what other
papers use.  `INSPIRE keeps it current
<https://inspirehep.net/literature?q=arxiv:2408.10926>`_ if the reference
changes.

Citing a specific version
-------------------------

The paper describes the pipeline, not any particular release.  If a result
depends on the version of the code that produced it — and for anything
touching the noise model it does, see :doc:`known_issues` — record the
version alongside the citation.  Files written by GRANDlib carry it in
``TRun.software_version``.

Citing the components
---------------------

GRANDlib orchestrates external packages that do most of the physics.  Work
that leans on them should cite them too: ZHAireS :cite:`Alvarez-Muniz:2010hbb`
or CoREAS :cite:`Huege:2013vt` for the air-shower simulation, DANTON
:cite:`Niess:2018opy` for :math:`\nu_\tau` propagation, and TURTLE
:cite:`Niess:2019hdn` for topography.  The full list is in
:doc:`references`.
