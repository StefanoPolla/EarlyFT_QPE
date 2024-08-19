"""
Logical costing of THC circuits.

While the functions in this module could, in principle, be used for any circuit, they are designed
and tested on THC walk operators only. 
TODO: generalize the functions and move them elsewhere in the codebase. Maybe use for Hubbard too.
"""
import numpy as np


def walk_call_graph(walk_operator):
    """
    Utility function that constructs a call graph adapted for CCZ counting.

    see `qualtran.Bloq.call_graph` for more details.
    """
    from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join

    # Define the generalizers
    def generalize_cliffords(b):
        """A generalizer that replaces Clifford bloqs with ArbitraryClifford."""
        from cirq import has_stabilizer_effect
        from qualtran.bloqs.bookkeeping import ArbitraryClifford
        from qualtran.resource_counting.classify_bloqs import bloq_is_clifford

        if bloq_is_clifford(b) or has_stabilizer_effect(b):
            return ArbitraryClifford(n=b.signature.n_qubits())
        return b

    def and_to_toffoli(b):
        """A generalizer that replaces And bloqs with Toffolis, useful with CCZ factories."""
        from qualtran.bloqs.basic_gates import Toffoli
        from qualtran.bloqs.mcmt import And

        if isinstance(b, (And)):
            if not b.uncompute:
                return Toffoli()
        return b

    def fix_equals_a_constant(b):
        """
        A generalizer that replaces EqualsAConstant with MultiControlX.
        currently the bloq EqualsAConstant does not define a decomposition, and its call graph
        is hardcoded as a number of T gates. This generalizer replaces it with a MultiControlX,
        which is equivalent and has a known Toffoli decomposition.
        """
        from qualtran import Adjoint
        from qualtran.bloqs.arithmetic.comparison import EqualsAConstant
        from qualtran.bloqs.mcmt import MultiControlX

        if isinstance(b, (EqualsAConstant)):
            return MultiControlX([1] * b.bitsize)
        if isinstance(b, (Adjoint)):
            if isinstance(b.subbloq, (EqualsAConstant)):
                return MultiControlX([1] * b.subbloq.bitsize)
        return b

    def keep(b):
        """
        As we want to count CCZ magic (i.e. Toffoli up to a clifford), we do not want Toffoli and
        Fredkin (TwoBitCSwap) to be decomposed to T gates.
        """
        from qualtran.bloqs.basic_gates import Toffoli, TwoBitCSwap

        if isinstance(b, (Toffoli, TwoBitCSwap)):
            return b

    g, sigma = walk_operator.call_graph(
        generalizer=[
            ignore_split_join,
            ignore_alloc_free,
            generalize_cliffords,
            and_to_toffoli,
            fix_equals_a_constant,
        ],
        keep=keep,
    )

    return g, sigma


def magic_from_sigma(sigma):
    """
    Computes CCZ+T magic from sigma, aggregating leaf bloq costs.

    Converts Fredkins (TwoBitCSwap) to Toffolis and rotations to T gates.
    """
    from cirq import has_stabilizer_effect
    from qualtran.bloqs.basic_gates import TGate, Toffoli, TwoBitCSwap
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity
    from qualtran.resource_counting.t_counts_from_sigma import _get_all_rotation_types
    from qualtran.surface_code import MagicCount

    rotation_types = _get_all_rotation_types()

    ccz = sigma.get(TGate(), 0)
    ccz += sigma.get(TwoBitCSwap(), 0) * 3  # Fredkin is 3 Toffolis
    tgates = sigma.get(TGate(), 0)
    # count rotations as T gates
    for bloq, counts in sigma.items():
        if isinstance(bloq, rotation_types) and not has_stabilizer_effect(bloq):
            tgates += np.ceil(TComplexity.rotation_cost(bloq.eps)) * counts

    return MagicCount(n_ccz=sigma.get(Toffoli(), 0), n_t=tgates)
