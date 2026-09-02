# Why CI stopped working

Diagnosed 2026-08-30 from run history and the GitHub API. Three independent
causes, which is why partial fixes would not have revived it.

## 1. `tests.yml` was disabled by hand

```
$ gh workflow list -R grand-mother/grand --all
  disabled_manually   Tests        .github/workflows/tests.yml
  disabled_manually   AppImage     .github/workflows/appimage.yml
  active              Tests with docker
  active              Versioning test
```

This is why the main quality gate has **zero** runs in its entire history —
not a trigger bug, not a path filter. Someone switched it off in the UI and it
was never switched back.

## 2. Its container no longer exists

`tests.yml` runs in `docker://jcolley/grand:0.4`. Docker Hub returns **404**
for that tag. Re-enabling the workflow as written would fail at container
setup.

`tests_with_docker.yml` uses `docker://jcolley/grandlib_ci:0.1`, which does
still exist: 872 MB, linux/amd64, **last pushed 2022-01-31**.

## 3. The surviving container is too old for the runner

`tests_with_docker.yml` is active and does start, but every run ends the same
way:

```
created: 2026-08-27T08:15:08Z
ended:   2026-08-28T08:15:10Z    <- 24 h 2 s later
conclusion: cancelled
job: Linux [cancelled] steps: (none)
```

Exactly the 24-hour job limit, with **no step ever recorded**. The job never
reached `actions/checkout`; it hung in container setup and was killed by the
timeout. That is the signature of a container the runner cannot operate in:
GitHub injects its own Node.js to execute JavaScript actions, and a
January-2022 image predates the glibc that current Node requires. The
workflow also still pins `actions/checkout@v2`.

## Consequences

- No merge in recent memory has been validated by anything.
- Appendix C of the paper reports 84% coverage. That was true when measured
  and has been unverifiable since.
- Fork pull requests (154, 150, 149) get no CI at all, because both workflows
  trigger on `push` only.

## What Phase 2 has to do

1. Build the container **from a Dockerfile in this repository**, published to
   GHCR under the `grand-mother` organisation. The present situation — a
   personal Docker Hub account, one image deleted and the other four years
   stale — is the root cause of two of the three failures.
2. Re-enable `tests.yml`, or replace it. It cannot simply be switched on: its
   image is gone.
3. Update the action versions; `actions/checkout@v2` is long out of support.
4. Add `pull_request` to the triggers.
5. Move the `paths:` filter off the triggers and into the workflow, so that a
   filtered commit still reports a check and can be required.
6. Add a job-level `timeout-minutes`. A hang should fail in ten minutes, not
   occupy a runner for a day.

A local environment that works now exists (`env/conda/grand-dev.yml`), so the
container can be built from the same specification the collaboration develops
against, which closes the people-versus-CI gap at the same time.
