# setup.py
from setuptools import setup, find_packages

dependencies = [
    "qualtran==0.4.1",
    "tqdm==4.66.4",
    "seaborn==0.13.2",
    "scipy==1.12.0",
    "pandas==2.2.2",
    "numpy==1.26.4",
    "matplotlib==3.9.0",
    "tables==3.10.1"
]


setup(
    name="eftqpe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=dependencies,
    author="Stefano Polla",
    author_email="polla@lorentz.leidenuniv.nl",
    description="Tools for Early Fault-tolerant Quantum Phase Estimation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/StefanoPolla/EarlyFT_QPE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
