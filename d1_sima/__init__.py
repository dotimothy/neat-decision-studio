"""LiquidAI's d1 decision models compiled for the SiMa.ai Modalix MLA.

d1-omni-600M is an LFM2 trunk made bidirectional (short convolutions and grouped-query
attention), a small decision head, and a SigLIP2 tower for pictures. Each is built here as an
MLA graph with the LLiMa compiler framework, as `laya_sima` and `clm_sima` build theirs.
"""
