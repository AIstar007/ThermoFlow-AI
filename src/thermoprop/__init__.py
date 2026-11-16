"""Thermal propagation analysis toolkit."""
from .analysis import AnalysisResult, analyze_sequence, rectangular_roi
from .pipeline import run_pipeline

__all__ = ["AnalysisResult", "analyze_sequence", "rectangular_roi", "run_pipeline"]
__version__ = "1.5.0"
