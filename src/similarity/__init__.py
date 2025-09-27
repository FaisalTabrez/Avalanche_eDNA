"""
Similarity analysis module for cross-analysis comparisons.

This module provides sophisticated algorithms for comparing analysis results
across different reports and calculating comprehensive similarity metrics.
"""

from .cross_analysis_engine import CrossAnalysisEngine, SimilarityCalculator

__all__ = [
    'CrossAnalysisEngine',
    'SimilarityCalculator'
]