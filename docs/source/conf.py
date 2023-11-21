# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pathlib import Path
import sys
sys.path.insert(0, Path(__file__).parents[2].resolve().as_posix())

project = 'Spotipy'
copyright = '2023, Albert Dominguez Mantes'
author = 'Albert Dominguez Mantes'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    "sphinx_immaterial",
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_exec_code',
    'sphinx_immaterial.apidoc.python.apigen',
]
templates_path = ['_templates']
exclude_patterns = []

python_apigen_modules = {
      "spotipy_torch": "api",
}
python_apigen_order_tiebreaker = "definition_order"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_immaterial'
html_static_path = ['_static']


html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    # "site_url": "https://jbms.github.io/sphinx-immaterial/",
    "repo_url": "https://github.com/weigertlab/spotipy-torch",
    "repo_name": "Spotipy",
    "edit_uri": "blob/main/docs",
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        "navigation.sections",
        "navigation.top",
        "search.share",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "light-green",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "deep-orange",
            "accent": "lime",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    "toc_title_is_page_title": True,
}
