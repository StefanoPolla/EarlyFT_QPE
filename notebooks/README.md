# Scripts and Notebooks
The scripts and notebooks in this directory are used to generate the data and
figures for the [manuscript arXiv:2410.05369](https://arxiv.org/abs/2410.05369).

## THC Costing

Since the THC cost estimates require some prerequisites, including (computationally expensive) factorization
of the molecular Hamiltonian through tensor hypercontraction (THC), the steps to generate all data are split up
in the `test_generate_thc_costing.py` script. This script can be run via `pytest`, i.e., it parametrizes the individual
functions over the different test cases. As such, intermediate data are saved even when something fails.

1. Generation of the THC factorizations for simple molecules: `pytest test_generate_thc_costing.py -k "test_generate_thc_npz"`
2. Generate QPE cost estimates: `pytest test_generate_thc_costing.py -k "test_estimate_qpe_resources"`
3. Use the notebook `costing_plots_manuscript.ipynb` to re-generate the costing plots from the manuscript.