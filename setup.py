'''
Date: 2023-01-25 19:36:50
LastEditTime: 2023-03-06 00:20:47
Description: 
'''

from setuptools import setup, find_packages

setup(name='safebench',
      packages=["safebench"],
      include_package_data=True,
      version='1.0.0',
      install_requires=['gym', 'pygame'])
