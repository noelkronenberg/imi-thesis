# configuration file for the Sphinx documentation builder (https://www.sphinx-doc.org/en/master/usage/configuration.html)

import os
import sys
sys.path.insert(0, os.path.abspath('../../corr_utils')) # location of package

# project information (https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information)

project = 'corr_utils'
copyright = '2024, Noel Kronenberg'
author = 'Noel Kronenberg'
release = '0.1'

# general configuration (https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # Google docstrings
    'sphinx.ext.viewcode', # adds links to highlighted source code
]

templates_path = ['_templates']

exclude_patterns = [
    '/corr_utils/tests/**',
]

# options for HTML output (https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output)

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']