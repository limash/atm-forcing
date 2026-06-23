# NORA3 data fetching and processing for ocean models.

## Installation

The regridding stack (ESMF and its Python bindings `esmpy`/`xesmf`) is **not**
available on PyPI, so it must come from conda-forge. Everything else is managed
by [uv](https://docs.astral.sh/uv/): conda provides the ESMF binaries, then uv
installs the project and its PyPI dependencies into the same environment.

1. Install [miniforge/conda](https://conda-forge.org/download/) and
   [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone and enter the repo:
   ```bash
   git clone https://github.com/limash/atm-forcing.git
   cd atm-forcing
   ```
3. Create a conda environment providing the ESMF regridding stack:
   ```bash
   conda create -n atm-forcing -c conda-forge python=3.12 esmf esmpy xesmf
   conda activate atm-forcing
   ```
4. With that env active, install the project and the remaining (PyPI)
   dependencies with uv:
   ```bash
   uv pip install -e .
   ```
   `uv pip install` automatically targets the activated conda env (via
   `CONDA_PREFIX`) — no environment variables to set — and, unlike `uv sync`,
   leaves the conda-provided `esmf`/`esmpy`/`xesmf` untouched. (Use a named env
   as above, not the conda `base` env, which uv treats as a system environment.)

> A single tool cannot install both the conda and PyPI pieces, because `esmpy`
> is conda-only and uv has no conda support. If you'd prefer one tool for
> everything, use [pixi](https://pixi.sh) instead of conda + uv.

## Usage
With the `atm-forcing` env active, run `python app/nora3.py -o where/to/save`.

This command downloads the NORA3 dataset and interpolates it to the lat–lon grid covering the Oslofjord region for the years 2010–2020.
**Note:** The process may take several days to complete.
You can adjust the spatial domain by modifying the settings in `nora3.py`.
