import os
import sys

sys.path.extend([os.path.dirname(os.getcwd()), os.path.join(os.path.dirname(os.getcwd()), "jaxparrow")])
autodoc_member_order = 'groupwise'
exclude_patterns = ['_build', '**tests**']


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'jaxparrow'
copyright = '2023, Victor Zaia, Vadim Bertrand, Emmanuel Cosme, Julien Le Sommer'
author = 'Victor Zaia, Vadim Bertrand, Emmanuel Cosme, Julien Le Sommer'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
