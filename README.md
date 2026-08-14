# PyMC with MLflow

These are simple examples of using PyMC with MLflow, taking advantage of the
`pymc_marketing.mlflow` module.

This focuses on logging parameters, metrics, and artifacts to MLflow.

The examples target `pymc-marketing>=1.0`, `pymc>=6`, `arviz>=1.2` (the new
ArviZ metapackage), and `mlflow>=3.15`. If you are migrating your own MMM code
from `pymc-marketing<1.0`, see the
[MMM migration guide](https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_migration_guide.html).

![Autologging](./images/autolog.png)

*(screenshot from pymc-marketing 0.15; the exact autologged surface differs slightly in 1.0)*

Suggestions or Questions? [Comment on this Issue](https://github.com/pymc-labs/pymc-marketing/issues/938)

## Environment (pixi)

The project is managed with [pixi](https://pixi.sh); the dependencies and the
multi-platform lockfile (`pixi.lock`, covering `osx-arm64`, `linux-64`, and
`osx-64`) live in `pyproject.toml`.

```bash
# install pixi once (see https://pixi.sh/latest/#installation)
curl -fsSL https://pixi.sh/install.sh | sh

# create the environment from the committed lockfile
pixi install
```

All `make` targets wrap their commands with `pixi run`, so there is no need to
activate anything — but they must be run **from the repository root**, since
the MLflow tracking database and artifact paths are relative.

## MLflow server

### Spin it up

```bash
make serve
```

This runs:

```bash
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --port 5001
```

- The UI is served at <http://127.0.0.1:5001>. The server binds to localhost
  only; add `--host 0.0.0.0` to expose it beyond your machine.
- Port 5001 is used instead of MLflow's default 5000 because macOS runs its
  AirPlay Receiver on port 5000, which answers `localhost` requests over IPv6
  with a 403 before MLflow's IPv4-only listener is reached.
- `mlruns.db` is a SQLite database acting as the tracking backend — the same
  file the example scripts write to directly (see `mlflow_set_tracking_uri`
  in [utils.py](./utils.py)), so you can run the experiments first and browse
  them afterwards.
- Artifacts (figures, inference data, models) are stored on the local
  filesystem at the location recorded when each experiment is created.

### Prune it (sandbox workflow)

Deleting runs in the UI (or with `mlflow experiments delete`) only
*soft-deletes* them — the rows and artifacts stay on disk. To reclaim space in
a throwaway/sandbox setup:

```bash
# 1. stop the server first: SQLite allows a single writer, and running gc
#    against a live server can fail with "database is locked"
# 2. hard-delete everything that was soft-deleted
make prune
```

`make prune` runs
`mlflow gc --tracking-uri sqlite:///mlruns.db --backend-store-uri sqlite:///mlruns.db`.
Useful variations (run them with `pixi run mlflow gc ...`):

- `--older-than 7d` — only garbage-collect runs deleted more than 7 days ago
- `--experiment-ids 1,2` — restrict to specific experiments
- `--run-ids <id>,<id>` — restrict to specific runs

For a complete reset of the sandbox (tracking database **and** all artifacts):

```bash
make clean_up   # rm -rf mlruns mlruns.db
```

## Scripts

There are four scripts:

1. [Non-PyMC example showing how to log parameters, metrics, and artifacts to MLflow](./01-basic-introduction.py)
2. [PyMC example which logs some PyMC related metrics to MLflow](./02-pymc-context.py)
3. [Logging that and more with `pymc_marketing.mlflow` module](./03-pymc-autologging.py)
4. [Autologging of Marketing Mix Model with `pymc_marketing.mlflow` module](./04-pymc-marketing-mmm)

There are some helper functions in the `utils.py` file which help setup mlflow and define some reused PyMC models.

## Running the experiments

You can either kick off everything at once with the batch script, or run a
single script at a time to explore one example.

No MLflow server needs to be running beforehand: the scripts write straight to
the `mlruns.db` SQLite file, so you can start the UI afterwards to browse the
results.

### All at once

```bash
make experiments
```

This wraps `pixi run bash ./kick-off.sh` — you can also call
`pixi run bash ./kick-off.sh` directly. It invokes script 01 four times,
02 twice, 03 six times across different sampler/likelihood combinations, and
04 once, which by itself fits the whole model grid. That comes out to 20 MLflow
runs spread over the four experiments.

Two things worth knowing:

- The `03-pymc-autologging.py pymc gamma` line is *meant* to fail — the
  generated data contains negative values, which a Gamma likelihood cannot
  accommodate. It is there so you can see what a failed run looks like in
  MLflow.
- The script does `export PYTHONPATH=.` before invoking 04 (see
  [the gotchas below](#gotchas)).

Since the whole batch fits real models, expect it to take a while. Stopping the
MLflow server before a large batch avoids SQLite writer contention.

### One script at a time

```bash
pixi run python 01-basic-introduction.py
pixi run python 02-pymc-context.py
pixi run python 03-pymc-autologging.py <nuts_sampler> <likelihood> [--mock]
PYTHONPATH=. pixi run python 04-pymc-marketing-mmm [--config <path>]
```

Scripts 01 and 02 take no arguments.

Script [03](./03-pymc-autologging.py) takes two positional arguments:

- `nuts_sampler`: one of `pymc`, `nutpie`, `numpyro`. Choosing `pymc`
  additionally attaches a callback that logs sampler stats every 100 draws.
- `likelihood`: one of `normal`, `student_t`, `gamma`.

It also accepts `--mock`, which swaps in `pymc.testing.mock_sample` instead of
actually sampling — handy for iterating on the logging code quickly:

```bash
pixi run python 03-pymc-autologging.py nutpie normal --mock
```

Script [04](./04-pymc-marketing-mmm) is run as a directory module and accepts
`--config`, defaulting to
[run-config.yaml](./04-pymc-marketing-mmm/run-config.yaml). It fits the product
of the `adstocks`, `saturations`, and `yearly_seasonality` entries in that file
(currently 2 x 2 x 2 = 8 models, one MLflow run each) and downloads the example
dataset over the network.

### Gotchas

- **Run everything from the repository root.** `mlflow_set_tracking_uri` in
  [utils.py](./utils.py) points at `sqlite:///mlruns.db`, a relative path, and
  artifacts are resolved relative to it as well.
- **`PYTHONPATH=.` is required for script 04.** Running a directory as a module
  puts `04-pymc-marketing-mmm/` on `sys.path` rather than the repository root,
  so the shared `from utils import ...` would otherwise fail.

Each script writes to its own MLflow experiment: `01-basis-introduction`,
`02-pymc-context`, `03-pymc-autologging`, and `04-pymc-marketing-mmm`.

View the results with `make serve` and reset the sandbox with `make clean_up`,
both covered below.

## Resources

- [`pymc_marketing.mlflow` module](https://www.pymc-marketing.io/en/latest/api/generated/pymc_marketing.mlflow.html)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [MLflow Tracking Server](https://mlflow.org/docs/latest/tracking/server.html)
