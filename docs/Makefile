# Minimal Sphinx makefile.
#
# `html` depends on the Handbook PDF, because the documentation links to it
# with :download: and Sphinx warns if the target is missing -- which would stop
# the build being warning-clean.  The recipe compiles it where pdflatex is
# available and falls back to the copy shipped in resources/ where it is not,
# so `make html` works either way.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

handbook-pdf:
	@python dev/build_handbook_pdf.py

# Depends on the phony target, not on the file: the catch-all %% rule below
# would otherwise claim the file name and hand it to sphinx-build as a builder.
html: handbook-pdf
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help handbook-pdf html Makefile

%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
