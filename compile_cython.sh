#!/bin/bash

CYTHON_FILES=$(find . -type f -name "*.pyx")
for file in $CYTHON_FILES; do
    cythonize -3 -i -a $file
done
