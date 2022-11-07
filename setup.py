import setuptools
from pathlib import Path

setuptools.setup(
    name="IMSE",
    version="1.0.0",
    packages=setuptools.find_packages(),
    url="https://github.com/sgibson-mse/IMSE",
    author="Sam Gibson",
    author_email="sam.gibson@ukaea.uk",
    description="Imaging Motional Stark effect analysis package",
    long_description=open(
        Path(__file__).parents[0] / "docs" / "README.md"
    ).read(),
    install_requires=['pycpf==0.1',
                      'Shapely==1.7.1',
                      'uda==2.5.0',
                      'uda-mast==1.3.3'
    ],
)