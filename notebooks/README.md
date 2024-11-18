# Scripts and Notebooks
The scripts and notebooks in this directory are used to generate the data and
figures for the [manuscript arXiv:2410.05369](https://arxiv.org/abs/2410.05369).

## THC Costing

To run the THC costing, please install this repository with "extras" options:
```bash
pip install .[extras]
```
This will additionally install PySCF, JAX, and the most recent version of OpenFermion (pulled from GitHub) to support
the newest JAX releases.

Since the THC cost estimates require some prerequisites, including (computationally expensive) factorization
of the molecular Hamiltonian through tensor hypercontraction (THC), the steps to generate all data are split up
in the `test_generate_thc_costing.py` script. This script can be run via `pytest`, i.e., it parametrizes the individual
functions over the different test cases. As such, intermediate data are saved even when something fails.

1. Generation of the THC factorizations for simple molecules: `pytest test_generate_thc_data.py -k "test_generate_thc_npz"`
2. Generate QPE cost estimates: `pytest test_generate_thc_costing.py`
3. Use the notebook `costing_plots_manuscript.ipynb` to re-generate the costing plots from the manuscript.