import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'PyIto'
copyright = '2025, Rohit Rawat' 
author = 'Rohit Rawat'        
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',      
    'sphinx.ext.napoleon',     
    'sphinx.ext.viewcode',    
    'sphinx.ext.mathjax',      
    'myst_parser',             
]

templates_path = ['_templates']
exclude_patterns = []
--
html_theme = 'sphinx_rtd_theme'  
html_static_path = ['_static']

autoclass_content = 'both'