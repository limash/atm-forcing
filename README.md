# NORA3 data fetching and processing for ocean models.

## Installation
1. Install conda <https://conda-forge.org/download/>.
2. Create a conda environment `conda create --name nora3 python=3.12`
3. Activate the envoronment `conda activate nora3`
4. `git clone https://github.com/limash/atm-forcing.git`
4. Navigate to the atm-forcing directory you git cloned
5. Install the dependencies `pip install -e .`
6. Xesmf is required, see <https://xesmf.readthedocs.io/en/stable/installation.html>;
   install with `conda install -c conda-forge xesmf`.

## Usage
Run `python app/nora3.py -o where/to/save`.

This command downloads the NORA3 dataset and interpolates it to the lat–lon grid covering the Oslofjord region for the years 2010–2020.
**Note:** The process may take several days to complete.
You can adjust the spatial domain by modifying the settings in `nora3.py`.
