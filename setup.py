"""
Setup file with config details for the package.
"""
from setuptools import setup

setuptools_kwargs = {
    "install_requires": [
        "pyomo",
        "matplotlib",
        "pandas",
        "networkx",
        "pytest",
    ],
    "python_requires": ">=3.7, <4",
}

setup(
    name="SNS",
    version="0.0.1",
    description="Pyomo models for synthesis of thermally coupled distillation columns",
    maintainer="Kevin Pfau",
    maintainer_email="kpfau@andrew.cmu.edu",
    license="BSD 3-Clause",
    long_description="""Package provides deterministic optimization models of thermally coupled
    separation networks for zeotropic mixtures formulated as a generalized disjunctive program. Models
    are built in the Pyomo modeling language with the Pyomo.GDP modeling extension""",
    **setuptools_kwargs
)
