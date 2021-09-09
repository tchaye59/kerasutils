from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("LICENSE", "r", encoding="utf-8") as f:
    LICENSE = f.read()

setup(
    name='kerasutils',
    version='0.0.0',
    packages=setuptools.find_packages(),
    url='https://github.com/tchaye59/kerasutils',
    license=LICENSE,
    author='Jude TCHAYE',
    author_email='tchaye59@gmail.com',
    description='My Keras models utils',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
