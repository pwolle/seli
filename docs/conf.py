"""Sphinx configuration file for the seli documentation."""

import os
import sys
from datetime import datetime

# Add the project root and src directories to the path
# so that autodoc can find the modules
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# --- Project information -------------------------------------------------------
project = "seli"
author = "Paul Wollenhaupt"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.1"  # Updated to match pyproject.toml


# --- HTML output configuration ------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "repository_url": "https://github.com/pwolle/seli",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "logo": {
        "text": "<span style='font-size: 2em;'>Seli</span>",
    },
}


html_title = "Seli Docs"

# static files
html_static_path = ["_static"]

html_context = {
    "default_mode": "dark",
}

html_css_files = [
    "custom.css",
]

# --- Extensions configuration -------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_toolbox.code",
    "sphinx_toolbox.github",
    "sphinx_automodapi.automodapi",
    "myst_parser",
]

# Autodoc configuration
autodoc_default_options = {
    "undoc-members": False,
}

github_username = "pwolle"
github_repository = "flarejax"
github_branch = "main"


# MyST-Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3
