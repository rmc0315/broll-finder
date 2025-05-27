#!/usr/bin/env python
"""Setup script for B-Roll Finder"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="broll-finder",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered tool to find B-roll footage for video scripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/broll-finder",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Content Creators",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "openai>=1.0.0",
        "nltk>=3.8.1",
        "spacy>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "broll-finder=main:main",
        ],
    },
)