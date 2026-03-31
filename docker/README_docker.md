# BAGLE Docker Environment

This directory provides a minimal Docker workflow for running BAGLE in an isolated, reproducible environment without changing the native installation path.

## What It Does

- Builds a Python 3.11 environment with pinned dependencies
- Builds MultiNest from source before installing `pymultinest`
- Installs BLAS/LAPACK system libraries required by MultiNest
- Installs the local BAGLE checkout into the container
- Exposes a writable `/data` mount for inputs and outputs
- Starts an interactive shell by default
- Uses the `bagle_env` Conda environment as the default runtime environment

## Files

- `Dockerfile`: container build definition
- `docker-compose.yml`: interactive local workflow
- `entrypoint.sh`: activates `bagle_env` for interactive shells and commands
- `test_imports.py`: smoke test for the container environment

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile -t bagle-in-docker:0.1 .
```

From the `docker/` directory:

```bash
docker build -f Dockerfile -t bagle-in-docker:0.1 ..
```

If you previously built an older image tagged `bagle-in-docker:0.1`, rebuild before testing again so Docker does not reuse the stale image.

## Run

### Option 1: Direct Docker

From the repository root:

```bash
mkdir -p data/input data/output data/test
docker run -it -v "$(pwd)/data:/data" bagle-in-docker:0.1
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

## Validate the Environment

Validated directly from the repository root with:

```bash
docker run --rm bagle-in-docker:0.1 python /opt/test_imports.py
```

Inside the container:

```bash
python /opt/test_imports.py
```

Expected output:

```text
Testing BAGLE Docker environment...
Core dependencies imported successfully.
BAGLE imported successfully.
Environment test PASSED.
```

## Notes

- Docker support is optional and does not replace the native BAGLE workflow.
- The image is CPU-only and does not include MPI, GPU, or HPC-specific support.
- The container installs the code from the current checkout, so local repo changes are reflected when you rebuild the image.
- MultiNest is built without MPI support in this MVP by design.
- The image now activates `bagle_env` explicitly at container startup, so both interactive shells and one-off commands use the BAGLE environment.
