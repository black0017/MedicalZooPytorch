#install sphinx
pip install sphinx
pip install sphinx-rtd-theme     #theme

# conf.py
**extensions:** 
'sphinx.ext.napoleon' for google style
sphinx.ext.autodoc for auto docs from docstrings

https://www.sphinx-doc.org/en/master/usage/extensions/index.html

# generate docs 
cd docs
sphinx-apidoc -f -o .  ../medzoo/

# generate html
make html

# resources
https://github.com/finsberg/sphinx-tutorial

#example
https://github.com/facebookresearch/mmf/tree/master/docs
https://mmf.sh/api/