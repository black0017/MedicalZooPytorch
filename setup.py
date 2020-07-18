import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="medzoo",
    version="1.0.0",
    description="",
    long_description=README,
    url="https://github.com/black0017/MedicalZooPytorch",
    author="",
    author_email="",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["medzoo"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "medzoo_train = medzoo_cli.train:train",
            "medzoo_predict = medzoo_cli.predict:predict",
        ]
    },
)
