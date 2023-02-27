from setuptools import find_packages, setup

setup(
    name="plumbing",
    version="0.1.0",
    description="template for quick data modelling",
    author="Ben Crittenden",
    author_email="ben.crittenden@babylonhealth.com",
    url="https://not-a-url.com",
    license="All rights reserved",
    py_modules=find_packages(),
    python_requires=">3.8.1",
    classifiers=[
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ],
)
