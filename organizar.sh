#!/bin/bash

# Crear carpetas si no existen
mkdir -p data
mkdir -p fig

# Mover archivos .csv y .dat
find . -maxdepth 1 -type f \( -name "*.csv" -o -name "*.dat" \) -exec mv -v {} data/ \;

# Mover archivos .png
find . -maxdepth 1 -type f -name "*.png" -exec mv -v {} fig/ \;
