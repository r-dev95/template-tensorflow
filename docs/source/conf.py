# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.append(os.path.abspath('../../src/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tensorflow-template'
copyright = '2024, r-dev95'
author = 'r-dev95'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    # 'myst_parser',
]

# sphinx.ext.autodoc
# autoclass_content = 'both' # [class, both, init]
autodoc_member_order = 'bysource' # [alphabetical, groupwise, bysource]
autodoc_typehints = 'none' # [signature, description, none, both]
# sphinx.ext.napoleon
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_custom_sections = [('Returns', 'params_style')]

# Options for internationalisation
language = 'en'
# Options for markup
keep_warnings = True
# Options for object signatures
toc_object_entries_show_parents = 'hide' # [domain, hide, all]
add_module_names = False
# Options for templating
templates_path = ['_templates']
# Options for source files
exclude_patterns = []
# Options for the nitpicky mode
nitpicky = True
nitpick_ignore_regex = [
    (r'py:class', r'Logger'),
    (r'py:class', r'logging.Formatter'),
    (r'py:class', r'Path'),
    (r'py:class', r'collections.abc.Callable'),
    (r'py:class', r'enum.StrEnum'),
    (r'py:class', r'pydantic.main.BaseModel'),
    (r'py:class', r'np.ndarray'),
    (r'py:class', r'tf.Tensor'),
    (r'py:class', r'tensorflow.python.framework.tensor.Tensor'),
    (r'py:class', r'tensorflow.core.example.feature_pb2.Feature'),
    (r'py:class', r'tensorflow.core.example.example_pb2.Example'),
    (r'py:class', r'keras.models.Model'),
    (r'py:class', r'keras.src.models.model.Model'),
]

gettext_compact = False
locale_dirs = ['locale/']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
