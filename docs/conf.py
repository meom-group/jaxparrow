import os
import sys

sys.path.extend([os.path.dirname(os.getcwd()), os.path.join(os.path.dirname(os.getcwd()), "jaxparrow")])
autodoc_member_order = "groupwise"
exclude_patterns = ["_build", "**tests**"]

from version import __version__


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "jaxparrow"
copyright = "2023, Victor E V Z De Almeida, Vadim Bertrand, Julien Le Sommer, Emmanuel Cosme"
author = "Victor E V Z De Almeida, Vadim Bertrand, Julien Le Sommer, Emmanuel Cosme"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
]
html_theme_options = {
    "description": "A JAX implementation of the cyclogeostrophic balance inversion",
    "github_user": "meom-group",
    "github_repo": "jaxparrow",
}
