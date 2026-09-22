# BAGLE Docker Environment

This directory provides a minimal Docker workflow for running BAGLE in an isolated, reproducible environment without changing the native installation path.

## What It Does

- Builds a Python 3.14 environment from BAGLE's `pyproject.toml`
- Builds MultiNest from source before installing `pymultinest`
- Installs BLAS/LAPACK system libraries required by MultiNest
- Installs BAGLE's JAX/PyMC dependencies from `pyproject.toml`, plus `jaxns`
  and its compatible `tfp-nightly` dependency required by BAGLE_time_tests
- Installs the local BAGLE checkout into the container
- Exposes writable `/data` and `/opt/BAGLE_time_tests` mounts for inputs,
  outputs, and the sibling BAGLE_time_tests checkout
- Exposes port `8888` for optional Jupyter access
- Starts an interactive shell by default
- Uses the `bagle_env` Conda environment as the default runtime environment

## Files

- `Dockerfile`: container build definition
- `docker-compose.yml`: interactive local workflow
- `test_imports.py`: smoke test for the container environment

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile -t bagle-in-docker:py314 .
```

From the `docker/` directory:

```bash
docker build -f Dockerfile -t bagle-in-docker:py314 ..
```

If you previously built an older image, rebuild before testing again so Docker
does not reuse the stale image.

## Run

### Option 1: Direct Docker

From the repository root:

```bash
mkdir -p data/input data/output data/test
docker run -it \
  -v "$(pwd)/data:/data" \
  -v "$(cd ../BAGLE_time_tests && pwd):/opt/BAGLE_time_tests" \
  bagle-in-docker:py314
```

### Option 2: Docker Compose

From the `docker/` directory:

```bash
mkdir -p ../data/input ../data/output ../data/test
docker compose up -d --build
docker exec -it bagle_container /bin/bash
```

If the container already exists, recreate it after a Dockerfile change:

```bash
docker compose down
docker compose up -d --build
```

Compose expects BAGLE_time_tests beside this repository. To use a checkout in a
different location, set `BAGLE_TIME_TESTS_PATH` to its absolute path before
starting Compose.

## JupyterLab

Jupyter support is optional and does not change the default CLI behavior.

From inside the running container:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

If you started the service with Docker Compose, port `8888` is already published to the host.

The image also registers the `bagle_env` kernel, which appears in Jupyter as:

```text
Python (bagle_env)
```

## Validate the Environment

Validated directly from the repository root with:

```bash
docker run --rm bagle-in-docker:py314 python /opt/test_imports.py
```

Inside the container:

```bash
python /opt/test_imports.py
```

Expected output:

```text
Testing BAGLE Docker environment...
Core and time-test dependencies imported successfully.
BAGLE imported successfully.
Environment test PASSED.
```

## Run BAGLE_time_tests

After building with Compose, the sibling BAGLE_time_tests checkout is available
at `/opt/BAGLE_time_tests`. Run a quick benchmark from the repository root:

```bash
docker compose -f docker/docker-compose.yml run --rm bagle \
  python scripts/run_comparison.py \
  --outdir reports/docker_smoke \
  --model pspl \
  --prior-mode narrow \
  --scenario fake_data1 \
  --model-cadence 10 \
  --only numpyro_nuts_grad \
  --nuts-draws 10 \
  --nuts-tune 10 \
  --nuts-chains 1
```

The time-test checkout is a bind mount, so benchmark reports written beneath
`reports/` persist on the host. The full-suite launchers may also be run from
inside the container, but their host-specific Conda activation lines should be
removed or bypassed because this image already activates `bagle_env`.

## Notes

- Docker support is optional and does not replace the native BAGLE workflow.
- The image is CPU-only and does not include MPI, GPU, or HPC-specific support.
- The container installs the code from the current checkout, so local repo changes are reflected when you rebuild the image.
- Changes to `requirements.txt` and `pyproject.toml` are picked up when you rebuild the image.
- MultiNest is built without MPI support in this MVP by design.
- The Python 3.14 image uses `tfp-nightly`, which `jaxns` requires. It does not
  install the stable `tensorflow-probability` distribution because the two
  distributions share module paths and conflict.
- Interactive shells still start in `bagle_env` via the root shell init files, while the default container command remains `/bin/bash`.
