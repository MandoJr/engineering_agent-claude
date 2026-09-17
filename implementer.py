"""Backward-compatible import for the structured patch implementer.

`CodingImplementer` remains available for integrations written against the
first release, but it now executes the same structured, diff-first protocol.
"""
from .structured_implementer import StructuredCodingImplementer

CodingImplementer = StructuredCodingImplementer

__all__ = ["CodingImplementer", "StructuredCodingImplementer"]
