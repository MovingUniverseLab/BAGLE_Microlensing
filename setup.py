from setuptools import setup, find_packages

setup(
    name='micromodel',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='none',
    description='micromodel package',
    long_description=open('README.md').read(),
    install_requires=[],
    url='TBD',
    author='Niranjan_Bhatia',
    author_email='ninjab@berkeley.edu'
    package_dir={'BAGLE': 'src/BAGLE'}
)
