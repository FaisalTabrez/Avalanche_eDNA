"""
Package setup for Avalanche eDNA pipeline.

Installing this package (pip install -e .) makes the `src` package importable
from anywhere without sys.path hacks, including inside Docker and during CI.
"""

from setuptools import setup, find_packages

setup(
    name="avalanche-edna",
    version="0.1.0",
    description="End-to-end eDNA biodiversity assessment pipeline",
    author="FaisalTabrez",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
    ],
)
