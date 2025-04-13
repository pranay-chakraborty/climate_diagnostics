import os
import sys
sys.path.insert(0, os.path.abspath('../../src')) # Point Sphinx to the source code

# -- Project information -----------------------------------------------------
project = 'Climate Diagnostics Toolkit'
copyright = '2025, Pranay Chakraborty, Adil Muhammed I. K.'
author = 'Pranay Chakraborty, Adil Muhammed I. K.'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'nbsphinx',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []
master_doc = 'index' # Explicitly set for older Sphinx versions if needed

# -- Options for Autodoc -----------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': False, # Set to True to find undocumented members
    'private-members': False,
    'special-members': '__init__',
    'show-inheritance': True,
}
autoclass_content = 'both' # Include docstrings from class and __init__

# -- Options for Napoleon ----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for Intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'cartopy': ('https://scitools.org.uk/cartopy/docs/latest/', None),
    'dask': ('https://docs.dask.org/en/latest/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None), # Added based on Trends.py
}
intersphinx_timeout = 5

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# html_logo = "_static/logo.png" # to add logo
#html_theme_options = {} 
html_show_sphinx = False
html_show_copyright = True

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_execute = 'never' # 'auto' or 'always' to execute notebooks
nbsphinx_allow_errors = True # Continue build on notebook errors