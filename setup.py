from setuptools import setup
import setuptools


long_description = ""
LICENSE = ""
with open("README.md", "r", encoding="utf-8") as f:
    try:
        long_description = f.read()
    except Exception as e:
        print(e)

with open("LICENSE", "r", encoding="utf-8") as f:
    try:
        LICENSE = f.read()
    except Exception as e:
        print(e)


setup(
    name='kerasutils',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/tchaye59/kerasutils',
    license=LICENSE,
    author='Jude TCHAYE',
    author_email='tchaye59@gmail.com',
    description='My Keras models utils',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
