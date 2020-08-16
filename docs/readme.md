
cd docs

# install sphinx
- pip install -r requirements.tzt

# conf.py
**extensions:** 
https://www.sphinx-doc.org/en/master/usage/extensions/index.html

# generate docs 
- sphinx-apidoc -f -o ./source/ ../medzoo/

# generate html
make html

# view docs
open in browser _build/html/index.html

# resources
https://github.com/finsberg/sphinx-tutorial

# example
- https://github.com/facebookresearch/mmf/tree/master/docs
- https://mmf.sh/api/