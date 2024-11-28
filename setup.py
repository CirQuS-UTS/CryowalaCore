#For creating the package
#Imports numpy, pandas, json, scipy
from setuptools import setup
setup(
    name='CryowalaCore',
    packages=['CryowalaCore'],
    description='The python backend for Cryowala. Functions in this code can be used as a standalone cryogenic wiring analysis tool, allowing for more detailed and powerful analyses.',
    version='0.1',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
    ]
)