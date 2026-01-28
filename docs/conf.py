"""Sphinx configuration for Structure-Constrained SINDy documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# Project information
project = "Structure-Constrained SINDy"
copyright = "2024, SC-SINDy Authors"
author = "SC-SINDy Authors"
release = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# Templates
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# MyST settings
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
