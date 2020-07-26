import pathlib
import platform
import os
import re
import codecs
from setuptools import setup, find_packages


def clean_html(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html).strip()
    return cleantext


# Single sourcing code from here:
# https://packaging.python.org/guides/single-sourcing-package-version/
# not used yet
def find_version(*file_paths):
    here = os.path.abspath(os.path.dirname(__file__))

    def read(*parts):
        with codecs.open(os.path.join(here, *parts), "r") as fp:
            return fp.read()

    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def fetch_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
        # https://stackoverflow.com/a/12982689
        readme = clean_html(readme)
    return readme


def fetch_requirements():
    requirements_file = "requirements.txt"

    if platform.system() == "Windows":
        DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")

    with open(requirements_file) as f:
        reqs = f.read()

    reqs = reqs.strip().split("\n")
    return reqs


HERE = pathlib.Path(__file__).parent

DISTNAME = "medzoo"
DESCRIPTION = "medzoo: A 3D multi-modal medical image segmentation library in PyTorch"

LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL ="https://github.com/black0017/MedicalZooPytorch"
AUTHOR = ""
AUTHOR_EMAIL = ""
LICENSE = "MIT"
DEPENDENCY_LINKS = []
REQUIREMENTS = (fetch_requirements())
EXCLUDES = ("contribute", "docker", "datasets", "examples", "figures", "installation", "manual", "notebooks", "results", "runs", "tests")
EXT_MODULES = []

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        url=URL,
        license=LICENSE,
        include_package_data=True,
        packages=find_packages(exclude=EXCLUDES),
        python_requires=">=3.6",
        ext_modules=EXT_MODULES,
        version="1.0.0",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        dependency_links=DEPENDENCY_LINKS,
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
        ],
        entry_points={
            "console_scripts": [
                "medzoo_train = medzoo_cli.train:train",
                "medzoo_predict = medzoo_cli.predict:predict",
            ]
        },
    )