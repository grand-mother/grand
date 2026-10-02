Recovery plan
=============

The recovery plan is the program of work that brought the GRANDlib repository
back to a state in which changes are tested, reviewed and released.  Before it,
the default branch (``master``) was over a thousand commits behind the branch
people actually used (``dev``), continuous integration no longer completed a
run, and three dozen branches waited unmerged.  The plan sets out, in phases,
how to rebuild a single working trunk without losing any of that work.

What it means for you
---------------------

* **Use** ``dev-next``.  It holds everything that was on ``dev``, plus the
  fixes, tests and documentation of the recovery.  New work goes there.
* ``dev`` **and** ``master`` **are frozen, not deleted.**  ``master`` is tagged
  ``archive/master-2025-03`` and ``dev`` ``archive/dev-2026-09``, so both stay
  recoverable from any clone.
* **When the plan reaches Phase 9,** ``dev-next`` becomes ``main``, the
  repository's default branch.  Nothing is deleted before Phase 10.

The phases
----------

=====  ========================================  ===========
Phase  Goal                                      Status
=====  ========================================  ===========
0      An integration branch, ``dev-next``       done
1      One environment file for everyone         done
2      Continuous integration that runs          done
3      Tests before new features                 done
4      Merge the queue of open branches          done
5      Settle the open decisions                 in progress
6      Separate input, processing and output     to do
7      Documentation                             done
8      Governance and repository weight          in progress
9      Promote ``dev-next`` to ``main``          to do
10     Clean up branches and old directories     to do
=====  ========================================  ===========

What remains
------------

Before the promotion (Phase 9):

* Announce the freeze of ``dev``, so that no new work lands there.
* Verify the installation on a clean machine.
* Decide whether files simulated before the Galactic-noise correction of
  7 September 2026 are reprocessed (:ref:`issue-galactic-noise-normalisation`).
* Decide whether a Docker image is published and maintained
  (:ref:`issue-docker-unmaintained`).
* Fix a first version number and its date.

After it:

* Phase 6: restructure the simulation so that its physics can be called on
  arrays, with configuration objects and ROOT confined to reading and writing
  files (:ref:`issue-import-requires-root`).
* Phase 10: salvage the remaining unique work on old branches, archive-tag and
  delete them, and remove ``src_outlib/``.

The full plan
-------------

The complete plan, with every task, the exit criteria for the promotion and
the decisions made along the way, is kept with the code in
`resources/dev/dev-next/RECOVERY_PLAN.md
<https://github.com/grand-mother/grand/blob/dev-next/resources/dev/dev-next/RECOVERY_PLAN.md>`_
and updated in the same commits as the work it describes.
