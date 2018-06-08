
from setuptools import setup, find_packages

setup(name='maboss',
    version="0.6.2",
    packages=find_packages(exclude=["test"]),
    author="Nicolas Levy",
    author_email="nicolaspierrelevy@gmail.com",
    description="A python and jupyter API for the MaBoSS software",
    install_requires = [
        "colomoto_jupyter",
        "pyparsing",
        "ipywidgets",
        "matplotlib",
        "pandas",
    ])
