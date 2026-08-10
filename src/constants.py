"""
constants.py

Shared tokenization-format constants, used across every script that reads
or writes the move-token corpus format.

These are FIXED -- part of the data format specification, not
configuration -- so they should never be changed, and should always be imported from
"""

START_TOKEN = "<|SOM|>"
RESULT_TOKENS = {"<|1-0|>", "<|0-1|>", "<|1/2-1/2|>", "<|*|>"}
