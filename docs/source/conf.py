# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Climate Diagnostics Toolkit'
copyright = '2025, Pranay Chakraborty, Adil Muhammed I. K.'
author = 'Pranay Chakraborty, Adil Muhammed I. K.'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Add project root to path
sys.path.insert(0, os.path.abspath('../../src'))  # Add src directory to path

# -- Project information -----------------------------------------------------
project = 'Climate Diagnostics Toolkit'
copyright = '2025, Pranay Chakraborty, Adil Muhammed I. K.'
author = 'Pranay Chakraborty, Adil Muhammed I. K.'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Generate API documentation from docstrings
    'sphinx.ext.viewcode',      # Add links to view source code
    'sphinx.ext.napoleon',      # Support NumPy and Google style docstrings
    'sphinx.ext.mathjax',       # Render math equations
    'sphinx.ext.intersphinx',   # Link to docs of other projects
    'sphinx.ext.autosummary',   # Generate summaries automatically
    'sphinx_autodoc_typehints', # Include type hints in the documentation
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Napoleon settings (for docstrings) -------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Intersphinx mapping ---------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'cartopy': ('https://scitools.org.uk/cartopy/docs/latest/', None),
    'dask': ('https://docs.dask.org/en/latest/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'titles_only': False,
    'logo_only': False,
}

