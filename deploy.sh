#!/usr/bin/env bash

# Stop on error
set -e

# Remove old build
rm -rf dist/*

# Build
python setup.py sdist bdist_wheel

# Check built files
twine check dist/*

# Upload
twine upload dist/*