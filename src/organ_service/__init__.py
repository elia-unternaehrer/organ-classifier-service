"""Organ classification service.

The package is split into a training half and a serving half. Both import
``organ_service.preprocessing`` so that the transform applied at training time
is byte-for-byte the transform applied at inference time.
"""

__version__ = "0.1.0"
