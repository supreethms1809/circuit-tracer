"""Scored auto-interpretability (REQ-4): detection + fuzzing scoring.

Protocol follows Paulo, Shabalin & Belrose: generate a natural-language
explanation per feature from max-activating contexts, then *score* it —
detection (does the scorer predict where the feature fires on held-out
contexts?) and fuzzing (can the scorer tell genuine from decoy token
highlights?). Naming alone does not answer the review; scoring does.

For naming features that appear in circuit graphs (e.g. RAVEL eval graphs),
use ``name_graphs`` / ``--feature-list`` so collect targets those IDs, then
writes explanations into each node's ``clerp`` field.
"""
