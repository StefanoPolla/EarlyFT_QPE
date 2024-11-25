import importlib
import numpy as np
import pytest
import pathlib

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.mcscf import avas
from openfermion.resource_estimates import thc
from openfermion.resource_estimates.molecule import ccsd_t

from eftqpe.thc_resources import run_thc

have_pybtas = importlib.util.find_spec("pybtas") is not None
thc_deps_available = thc.HAVE_DEPS_FOR_RESOURCE_ESTIMATES and have_pybtas

molecules = {
    "h2o": """O          0.000000000000     0.000000000000    -0.068516219310 
H          0.400000000000    -0.790689573744     0.543701060724 
H          0.000000000000     0.790689573744     0.543701060724""",
    "n2": """N 0 0 0
N 1.09 0 0
    """,
    "naphthalene": """C         16.71880      -11.96200        0.00000
C         16.71880      -10.28790        0.00000
C         18.16780      -12.77440        0.00000
C         18.16780       -9.47550        0.00000
C         15.26970      -12.77430        0.00000
C         15.26970       -9.47560        0.00000
C         19.59640      -11.94800        0.00000
C         19.59640      -10.30180       -0.00000
C         13.84110      -11.94800       -0.00000
C         13.84110      -10.30190        0.00000
H         18.19610      -14.06110        0.00000
H         18.19610       -8.18890       -0.00000
H         15.24140      -14.06110       -0.00000
H         15.24130       -8.18890        0.00000
H         20.70900      -12.59050       -0.00000
H         20.70890       -9.65920       -0.00000
H         12.72850      -12.59050       -0.00000
H         12.72860       -9.65930        0.00000""",
    "anthracene": """C      3.6609      0.5848      0.0000
  C      3.6110     -0.8397      0.0000
  C      2.4165     -1.4895      0.0000
  C      1.1870     -0.7528      0.0000
  C      2.5148      1.3166      0.0000
  C      1.2368      0.6679      0.0000
  C     -0.0491     -1.4032      0.0000
  C     -1.2369     -0.6678      0.0000
  C      0.0492      1.4033      0.0000
  C     -1.1871      0.7528      0.0000
  C     -2.5148     -1.3167      0.0000
  C     -3.6609     -0.5848      0.0000
  C     -3.6110      0.8395      0.0000
  C     -2.4165      1.4895      0.0000
  H      4.6397      1.0755      0.0000
  H      4.5529     -1.3980      0.0000
  H      2.3680     -2.5843      0.0000
  H      2.5432      2.4122      0.0000
  H     -0.0876     -2.4995      0.0000
  H      0.0876      2.4996      0.0000
  H     -2.5431     -2.4122      0.0000
  H     -4.6397     -1.0756      0.0000
  H     -4.5531      1.3975      0.0000
  H     -2.3682      2.5844      0.0000""",
}


@pytest.mark.skipif(
    not thc_deps_available,
    reason="THC dependencies (jax, pyscf, pybtas) not installed."
)
@pytest.mark.parametrize(
    "molname, ncas, nalpha, nbeta, ao_labels, nthc, penalty_param", [
        ("h2o", 6, 4, 4, ["H 1s", "O 2s", "O 2p"], 30, 1e-4),
        ("n2", 6, 3, 3, None, 30, 1e-4),
        ("naphthalene", 10, 5, 5, ["C 2pz"], 45, 1e-5),
        ("anthracene", 14, 7, 7, ["C 2pz"], 80, 5e-5),
    ]
)
def test_generate_thc_npz(molname, ncas, nalpha, nbeta, ao_labels, nthc, penalty_param, out):
    mf = gto.M(atom=molecules[molname], basis="cc-pvdz", verbose=4).apply(scf.RHF).run()
    if ao_labels is not None:
        norb, ne_act, mo_coeff = avas.avas(mf, ao_labels)
        assert norb == ncas
        assert ne_act == nalpha + nbeta
    else:
        mo_coeff = mf.mo_coeff
    # generate integrals
    mc = mcscf.CASCI(mf, ncas, (nalpha, nbeta))
    h1, core_energy = mc.get_h1eff(mo_coeff=mo_coeff)
    h2 = ao2mo.restore(1, mc.get_h2eff(mo_coeff=mo_coeff), ncas)

    ret = run_thc(
        h1, h2, core_energy=core_energy,
        nthc=nthc,
        symm_shift=True,
        verify=True,
        penalty_param=penalty_param,
    )
    ret.nalpha = nalpha
    ret.nbeta = nbeta
    # CCSD only because pt correction does not work with shifted integrals
    _, e_corr_ref, _ = ccsd_t(
        h1, eri=h2, ecore=core_energy, num_alpha=mc.nelecas[0], num_beta=mc.nelecas[1],
        use_kernel=False, no_triples=True
    )
    _, e_corr, _ = ccsd_t(
        h1, eri=ret.eri_reconstructed, ecore=core_energy, num_alpha=mc.nelecas[0], num_beta=mc.nelecas[1],
        use_kernel=False, no_triples=True
    )
    diff = np.abs(e_corr_ref - e_corr)
    print(ret.error_reconstruction, ret.lambda_thc, diff)
    ret.error_ccsd = diff
    
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    ret.to_npz(f"{out}/{molname}_thc_{ncas}_{nalpha}_{nbeta}_{nthc}.npz")


@pytest.mark.parametrize(
    "molname, integral_file, nthc, nalpha, nbeta", [
        ("a-S12-large", "data/thc/integrals_a-S12_26_27.npz", 100, 14, 13),
    ]
)
def test_generate_thc_from_file(molname, integral_file, nthc, nalpha, nbeta, out):
    integral_file = np.load(integral_file)
    h1 = integral_file["h1"]
    h2 = integral_file["h2"]
    core_energy = integral_file["core_energy"]
    ncas = h2.shape[0]

    _, e_corr_ref, _ = ccsd_t(
        h1, eri=h2, ecore=core_energy, num_alpha=nalpha, num_beta=nbeta,
        use_kernel=False, no_triples=True
    )

    penalty_param = 1e-5

    ret = run_thc(
        h1, h2, core_energy=core_energy,
        nthc=nthc,
        symm_shift=True,
        verify=True,
        penalty_param=penalty_param,
    )
    ret.nalpha = nalpha
    ret.nbeta = nbeta
    # CCSD only because pt correction does not work with shifted integrals
    _, e_corr, _ = ccsd_t(
        h1, eri=ret.eri_reconstructed, ecore=core_energy, num_alpha=nalpha, num_beta=nbeta,
        use_kernel=False, no_triples=True
    )
    diff = np.abs(e_corr_ref - e_corr)
    print(ret.error_reconstruction, ret.lambda_thc, diff)
    assert diff < 1e-3, f"CCSD error too large: {diff}. Please rerun."
    ret.error_ccsd = diff
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    ret.to_npz(f"{out}/{molname}_thc_{ncas}_{nalpha}_{nbeta}_{nthc}.npz")
