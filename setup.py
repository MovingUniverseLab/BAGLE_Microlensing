from setuptools import setup, find_packages
import re

PACKAGE_NAME = 'BAGLE'
SOURCE_DIRECTORY = 'src'
SOURCE_PACKAGE_REGEX = re.compile(rf'^{SOURCE_DIRECTORY}')

source_packages = find_packages(include=[SOURCE_DIRECTORY, f'{SOURCE_DIRECTORY}.*'])
proj_packages = [SOURCE_PACKAGE_REGEX.sub(PACKAGE_NAME, name) for name in source_packages]

setup(
    name=PACKAGE_NAME,
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='none',
    description='BAGLE: Bayesian Analysis of Gravitational Lensing Events',
    long_description=open('README.md').read(),
    install_requires=[],
    url='TBD',
    author='Niranjan_Bhatia, Jessica Lu',
    author_email='ninjab@berkeley.edu,jlu.astro@berkeley.edu',
    package_dir={'BAGLE': 'src'}
)
