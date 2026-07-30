"""
Project-wide constants that do NOT leak any private datasets.

Keep this file minimal and stable. Anything data-specific should be passed in
from the caller (e.g., id2tag mapping for NPS).
"""

BIO_OUTSIDE = "O"
