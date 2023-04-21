from typing import List
from setuptools import find_packages, setup
from os import path

HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = list()
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[i.replace('\n','') for i in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        if '' in requirements:
            requirements= [i for i in requirements if i != '']
        return requirements

setup (
    name='RegressorProject',
    version='0.0.1',
    author='Viresh',
    author_email='viresh.raj.sah@gmail.com',
    install_requires=get_requirements('C:\Users\U1143589\Learning\iNeuron\ML\End2end_project\requirements.txt'),
    packages=find_packages()
)