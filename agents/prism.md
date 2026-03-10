# Prism Agent

Prism is a multi-team LangGraph patching agent ported from `atlantis-crete`.

It iterates through three stages:

1. `evaluation_team` summarizes the failure and patch outcome.
2. `analysis_team` builds a notebook of relevant code and a fix strategy.
3. `patch_team` generates and reviews candidate patches.

The standalone `crs-prism` package wires this workflow to oss-crs through
`agents/prism.py`, `patcher.py`, and the libCRS-backed evaluator/environment.
