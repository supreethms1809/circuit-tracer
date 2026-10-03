"""Original-pipeline baselines — true AutoCircuit ACDC / Syed EAP.

Method IDs:
- ``eap_edge`` — Syed et al. Edge Attribution Patching (``auto-circuit``)
- ``acdc_edge`` — Conmy et al. ACDC Algorithm 1 (``auto-circuit``)

Outputs belong in ``macag_original_baselines.json``. Requires
``pip install auto-circuit`` (TL 3.x supported via ``tl_compat``).
"""

from macag.baselines.original.acdc_edge import run_acdc_edge
from macag.baselines.original.eap_edge import run_eap_edge
from macag.baselines.original.types import OriginalCircuitResult

__all__ = [
    "OriginalCircuitResult",
    "run_acdc_edge",
    "run_eap_edge",
]
