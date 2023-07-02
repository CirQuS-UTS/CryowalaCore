#For creating the package
#Imports numpy, pandas, json
from setuptools import setup
setup(
    name='Cryo_UTS',
    packages=['Cryo_UTS'],
    description='For modelling heat and noise loads',
    version='0.1',
    install_requires=[
        'numpy',
        'pandas',
    ]
)