from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requires = f.readlines()
    
install_requires = [req.strip() for req in requires]

setup(
    name = 'bagle',
    version = '0.1',
    license = 'GPLv3+',
    description = 'BAGLE: Bayesian Analysis of Gravitational Lensing Events',
    long_description = open('README.md').read(),
    url = 'https://github.com/MovingUniverseLab/BAGLE_Microlensing',
    author = 'Niranjan_Bhatia, Jessica Lu',
    author_email = 'jlu.astro@berkeley.edu',
    package_dir = {'': 'src'},
    packages = find_packages(where='src',
                           exclude=['tests*']),
    python_requires = '>=3.6',
    tests_require = ['pytest'],
    install_requires = install_requires
)
