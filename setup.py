from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy', 'Pillow==5.0.0', 'requests']

setup(
    name='monuments',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)
