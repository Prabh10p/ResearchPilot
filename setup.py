from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as obj_file:
        requirements = obj_file.readlines()
        requirements = [i.strip() for i in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name='ResearchPilot',  
    version='0.0.1',
    author='Prabhjot Singh',
    author_email='prabhjot.singh.data@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('Requirements.txt')
)