# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Kithara'
copyright = '2024, The Kithara Authors'
author = 'kithara_authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_permalinks_icon = '<span>#</span>'

html_theme_options = {
    "main_nav_links": {
      "GitHub": "https://github.com/AI-Hypercomputer/kithara",
    },
    "logo_light": "../images/kithara_logo_no_text.png",
    "logo_dark": "../images/kithara_logo_no_text.png",
    "show_prev_next": True,
}
