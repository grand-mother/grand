Recovery plan
=============

Where this repository is going, and why it needed a plan at all.

This page renders `docs/dev/RECOVERY_PLAN.md
<https://github.com/grand-mother/grand/blob/dev-next/docs/dev/RECOVERY_PLAN.md>`_
directly, so there is one source of truth, updated in the same commits as the
work it describes.  It is a **working document**: the unticked boxes are what
has not been done yet, and they are meant to be visible.

.. note::

   The short version, for anyone who does not want the whole plan.  ``master``
   is the GitHub default and sits over a thousand commits behind ``dev``, which
   is the real trunk; a second abandoned trunk, ``main``, was created in 2023.
   CI had not completed a run in a long time, so no recent merge was validated
   by anything, and thirty-six branches accumulated behind that.

   ``dev-next`` is the integration branch where the repair happens.  **Nothing
   is deleted before Phase 10** — keeping ``dev``, ``master`` and every branch
   intact through the transition is what makes rollback trivial.

   :doc:`contributing` is the practical companion: what the conventions are and
   how to run the checks.

----

.. include:: ../dev/RECOVERY_PLAN.md
   :parser: myst
