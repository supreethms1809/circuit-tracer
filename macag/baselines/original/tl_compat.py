"""TransformerLens 3.x compatibility for ``auto-circuit``.

``auto-circuit`` still imports ``transformer_lens.past_key_value_caching.HookedTransformerKeyValueCache``,
which TL 3 renamed to ``transformer_lens.cache.key_value_cache.TransformerLensKeyValueCache``.

Call :func:`enable_autocircuit_tl3_compat` **before** importing any ``auto_circuit`` module.
"""

from __future__ import annotations

import sys
import types


_ENABLED = False


def enable_autocircuit_tl3_compat() -> None:
    """Install import aliases so ``auto-circuit`` works with TransformerLens ≥ 3."""
    global _ENABLED
    if _ENABLED:
        return

    import transformer_lens
    import transformer_lens.cache.key_value_cache as kv

    cache_cls = kv.TransformerLensKeyValueCache
    entry_cls = getattr(kv, "TransformerLensKeyValueCacheEntry", None)

    mod = types.ModuleType("transformer_lens.past_key_value_caching")
    mod.HookedTransformerKeyValueCache = cache_cls  # type: ignore[attr-defined]
    if entry_cls is not None:
        mod.HookedTransformerKeyValueCacheEntry = entry_cls  # type: ignore[attr-defined]
    sys.modules["transformer_lens.past_key_value_caching"] = mod

    # Some auto-circuit modules do: from transformer_lens import HookedTransformerKeyValueCache
    transformer_lens.HookedTransformerKeyValueCache = cache_cls  # type: ignore[attr-defined]
    transformer_lens.past_key_value_caching = mod  # type: ignore[attr-defined]
    _ENABLED = True
