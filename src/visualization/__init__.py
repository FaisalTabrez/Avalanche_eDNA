"""
Visualization modules for biodiversity analysis
"""

try:
    from .dashboard import BiodiversityDashboard
except ImportError:
    try:
        from src.ui.dashboard import BiodiversityDashboard
    except ImportError:
        BiodiversityDashboard = None

from .plots import BiodiversityPlotter

__all__ = [
    'BiodiversityPlotter'
]
if BiodiversityDashboard is not None:
    __all__.append('BiodiversityDashboard')