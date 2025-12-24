import os
import sys

# 1. Point Sphinx to your source code (Go up 2 levels from docs/source)
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'PyIto'
copyright = '2025, Rohit Rawat'
author = 'Rohit Rawat'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# This list is CRITICAL. It tells Sphinx which plugins to load.
extensions = [
    'sphinx.ext.autodoc',      # Generates docs from code
    'sphinx.ext.napoleon',     # Parses NumPy style docstrings
    'sphinx.ext.viewcode',     # Links to source code
    'sphinx.ext.mathjax',      # Renders math equations
    'myst_parser',             # Parses Markdown (.md)
    'sphinx_rtd_theme',        # The theme
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Mock imports to avoid installing heavy dependencies on ReadTheDocs
autodoc_mock_imports = ["numba", "numpy"]