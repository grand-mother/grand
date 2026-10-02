Getting help
============

Questions, bug reports and requests all go to GRANDlib's GitHub repository.

Before you ask
--------------

#. Search the `open and closed issues
   <https://github.com/grand-mother/grand/issues?q=is%3Aissue>`_: the answer
   may already be there.
#. Check :doc:`known_issues` and :doc:`troubleshooting`.  Most errors start
   with ``GRANDlib:`` and the name of the function that raised them;
   searching these pages for that name usually finds them.

Opening an issue
----------------

`Open a new issue <https://github.com/grand-mother/grand/issues/new/choose>`_
and choose the template that fits: a bug report, a documentation problem, or
a blank issue for a question or a request.  A GitHub account is needed.

An issue gets answered fastest when it includes:

* **The shortest code that shows the problem**, with the file it reads if
  that file is in the repository, or a description of it if not.
* **The full output**, including the whole error message.  If the result is
  ``nan`` or a number you did not expect rather than an error, say what you
  expected and why.
* **Your versions**, from this command:

  .. code-block:: bash

      python -c "import sys, ROOT; from grand import provenance; print(provenance.current(), sys.version.split()[0], ROOT.gROOT.GetVersion())"

  which prints, for example,
  ``0.1.0.dev1 (git dev-next 5de0118f..., modified) 3.12.14 6.36.04``.

For a problem with a documentation page, the **Report a problem with this
page** link at the top of every page opens an issue with the page filled in.

Contributing a fix
------------------

If you can fix it yourself, a pull request is welcome: :doc:`contributing`
explains how.
