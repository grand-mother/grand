# Documentation sources

The documentation is the Sphinx site in `source/`; `dev/` holds the scripts
that generate parts of it (the Handbook pages, the PDF).  Build it with

```bash
cd docs && make html
```

and open `build/html/index.html`.  `algo/` holds older design notes, such as
[how the antenna response is computed](algo/Leff_processing.md).
