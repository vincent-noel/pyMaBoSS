
from setuptools import setup, find_packages

setup(name='maboss',
    version="0.6.4",
    packages=find_packages(exclude=["test"]),
    py_modules = ["maboss_setup"],
    author="Nicolas Levy",
    author_email="nicolaspierrelevy@gmail.com",
    description="A python and jupyter API for the MaBoSS software",
    install_requires = [
        "colomoto_jupyter >=0.4.10",
        "pyparsing",
        "ipywidgets",
        "matplotlib",
        "pandas",
    ])
