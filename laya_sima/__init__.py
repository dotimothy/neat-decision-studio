"""Laya (ModernBERT decision model) on SiMa.ai Modalix, built on the LLiMa compiler framework.

Import surface is deliberately thin: `hostio` needs only numpy, so the same pre/post
processing runs in the model-compiler venv, in the reference venv and in the DevKit tests.
"""
