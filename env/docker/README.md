# A generated GRANDlib image

`grandlib.dockerfile` builds an environment image from this repository's own
metadata: ROOT 6.36 from the upstream `rootproject/root` image, and the Python
dependencies that `pyproject.toml` declares.

It exists because `env/docker_amd64/` and `env/docker_arm64/` carry a
`requirements.txt` of 53 unpinned package names — a second dependency list that
has to be kept in step with the conda environment by hand, and was not: it was
the only place declaring `numba` and `lmfit`, and it drifted for three years
because nothing built it.

Here there is one source of truth, `pyproject.toml`, and two ways to deliver
it — conda for developers, this image for reproducibility.

```bash
docker build -f env/docker/grandlib.dockerfile -t grandlib:dev .
docker run --rm -it -v "$PWD:/opt/grandlib" grandlib:dev
# inside:
source env/setup.sh && pytest tests/ -q
```

The image does not contain the ~1 GB data model. `env/setup.sh` fetches it and
skips when it is already current, so baking it in would quadruple the image to
freeze something on its own release schedule.

**Status.** This is a proposal, not the supported route. Whether GRANDlib
supports Docker at all is an open question — see `issue-docker-unmaintained` in
the documentation. Nothing publishes this image to a registry; the CI job
`build-modern` in `.github/workflows/docker.yml` builds it and runs the suite
inside it, which is enough to know whether it works.
