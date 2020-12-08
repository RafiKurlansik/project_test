# Databricks notebook source
## Databricks notebook source

import setuptools
from setuptools import find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()
setuptools.setup(
     name='transformers',  
     version='0.1',
     author='Ricardo Portilla',
     author_email='ricodito@gmail.com',
     description='A Docker and AWS utility package',
     long_description=long_description,
   long_description_content_type='text/markdown',
     url='https://github.com/javatechy/dokr',
     packages=find_packages(),
     classifiers=[
         'Programming Language :: Python :: 3',
         'License :: OSI Approved :: MIT License',
         'Operating System :: OS Independent',
     ],
 )