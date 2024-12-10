# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

project = 'SIMSSNR'
copyright = '2024, Valerii Brudanin (TU Delft)'
author = 'Valerii Brudanin (TU Delft)'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram'
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True
}

def setup(app):
    # Connect the hook to Sphinx's autodoc process
    app.connect("autodoc-process-docstring", add_inheritance_diagram)
    app.connect("autodoc-process-docstring", add_module_inheritance_diagram)

def add_module_inheritance_diagram(app, what, name, obj, options, lines):
    """
    Automatically add a module-level inheritance diagram showing all classes.

    Args:
        app: The Sphinx application object.
        what: The type of object being documented ('module', 'class', etc.).
        name: The fully qualified name of the object being documented.
        obj: The object itself (e.g., a module object).
        options: The options for the autodoc directive.
        lines: The current docstring lines (this will be modified).
    """
    # Only act on modules (not individual classes)
    if what == "module":
        # Append a single inheritance-diagram directive for the module
        lines += [
            "",
            f".. inheritance-diagram:: {name}",
            "   :parts: 1",  # Includes only directly connected classes
            ""
        ]

def add_inheritance_diagram(app, what, name, obj, options, lines):
    """
    Automatically add inheritance diagrams to all class docstrings.

    Args:
        app: The Sphinx application object.
        what: The type of object being documented (e.g., 'class').
        name: The full name of the object being documented.
        obj: The object itself (e.g., a class object).
        options: Options for the autodoc directive.
        lines: The lines of the docstring (modifiable).
    """
    if what == "class" and hasattr(obj, "__bases__") and obj.__bases__:
        # Inject the inheritance diagram directive into the docstring
        lines += [
            "",
            ".. inheritance-diagram:: {}".format(name),
            "   :parts: 1",
            ""
        ]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'haiku'
html_static_path = ['_static']
