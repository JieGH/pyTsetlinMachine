from setuptools import *

libTM = Extension('libTM',
  ['pyTsetlinMachine/ConvolutionalTsetlinMachine.c', 'pyTsetlinMachine/MultiClassConvolutionalTsetlinMachine.c', 'pyTsetlinMachine/Tools.c'])

setup(
   name='pyTsetlinMachine',
   version='0.1.1',
   author='Ole-Christoffer Granmo',
   author_email='ole.granmo@uia.no',
   url='https://github.com/cair/pyTsetlinMachine/',
   license='MIT',
   description='Implements the Tsetlin Machine, Convolutional Tsetlin Machine and Regression Tsetlin Machine',
   ext_modules = [libTM],
   packages=['pyTsetlinMachine'],
)
