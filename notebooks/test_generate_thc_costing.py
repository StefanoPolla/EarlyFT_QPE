import numpy as np
import pytest
import pathlib


def get_npz_files(out):
    import glob
    npzs = glob.glob(f"{out}/*.npz")
    nacts = [int(npz.split("_")[-4]) for npz in npzs]
    npzs = [x for _, x in sorted(zip(nacts, npzs))]
    print(npzs)
    return npzs


@pytest.mark.parametrize("npz", get_npz_files("data/thc"))
def test_estimate_qpe_resources(npz, out):
    from eftqpe.thc_resources import estimate_thc_resources, thc_ftqc_physical_cost
    import pandas as pd
    
    nact = int(npz.split("_")[-4])
    if nact > 30:
        pytest.skip("Qubit counting takes too long.")

    gammas = np.logspace(-8, -1, 10)
    delta_e = 1e-3
    ftqc_error_budget = [1e-2, 1e-3]
    dfs = []
    ftqc_costs = []
    print('running', npz)
    mol = npz.split("/")[-1].split("_thc")[0]
    outdir = f"{out}/resources"
    fout = f"{outdir}/{mol}.h5"
    if pathlib.Path(fout).exists():
        pytest.skip(f"File {fout} exists.")
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    for n_factories in [1, 2]:
        df = estimate_thc_resources(thc_file=npz, delta_e=delta_e, n_factories=n_factories, gammas=gammas)
        df["molecule"] = mol
        dfs.append(df)

        total_qubits = df["total_qubits"].unique()[0]
        lambda_thc = df["lambda_thc"].unique()[0]
        ccz_count = df["magic_per_walk"].unique()[0].n_ccz
        for error_budget in ftqc_error_budget:
            cost_dict = thc_ftqc_physical_cost(
                ccz_count, total_qubits, lambda_thc,
                error_budget=error_budget, delta_e=delta_e, n_factories=n_factories,
            )
            # remove unnecessary keys
            cost_dict.pop("physical_cost", None)
            cost_dict.pop("factory", None)
            cost_dict.pop("data_block", None)
            # add input parameters
            cost_dict.update(
                {"n_factories": n_factories, "error_budget": error_budget, "molecule": mol,
                 "total_qubits": total_qubits, "lambda_thc": lambda_thc, "ccz_count": ccz_count}
            )
            ftqc_costs.append(cost_dict)

    df = pd.concat(dfs, ignore_index=True)
    df.drop(columns=["physical_cost", "factory", "data_block"], inplace=True)
    df.to_hdf(fout, key="df")

    df_ftqc = pd.DataFrame(ftqc_costs)
    df_ftqc.to_hdf(fout, key="df_ftqc")
