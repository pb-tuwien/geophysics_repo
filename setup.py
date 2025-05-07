from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Geophysics_Package",
    version="0.1",
    install_requires=requirements,  # Use the requirements from requirements.txt
    description="A package for working with Geophysical Data",
    author="Peter Balogh, Jakob Welkens"
)