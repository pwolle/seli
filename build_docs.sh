#!/bin/bash
# Script to clean, build and open the documentation

cd "$(dirname "$0")/docs"
make clean && make html && open _build/html/index.html
