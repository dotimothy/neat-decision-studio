"""CLM's encoder (Qwen3-8B, used for its last token's hidden state) on SiMa.ai Modalix, built
on the LLiMa compiler framework the way `laya_sima` builds Laya.

A decoder of this size does not go to the MLA as one graph: it is cut into consecutive runs of
layers, each its own graph and ELF, and the board runs them one after another.
"""
