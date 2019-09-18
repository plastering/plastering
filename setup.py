#!/usr/bin/env python
from pkg_resources import parse_requirements
from setuptools import setup, find_packages

__author__ = 'Jason Koh'
__version__ = '0.0.1'

install_reqs = parse_requirements(open('requirements.txt'))

reqs = [ir.name for ir in install_reqs]

setup(
    name = 'plastering',
    author = __author__,
    version = __version__,
    packages = find_packages() + ['config'],
    description = '[Plastering](https://github.com/plastering/plastering)',
    install_requires = reqs,
    #package_data = {
    #    'testtt': ['config/unit_mapping.csv', 'config/bacnettype_mapping.csv'],
    #}
    include_package_data = True,
)
