# script_puntos.gp

set datafile separator ","
set title "Solución Teórica: V(x, y) = (x - y)^2"
set xlabel "x"
set ylabel "y"
set zlabel "V_teorica"
set hidden3d
set style data points
set pointsize 0.5
set terminal pngcairo size 800,600 enhanced font 'Arial,10'
set output 'solucion_teorica_puntos.png'

splot 'solucion_teorica.csv' using 1:2:3 with points pt 7 lc rgb "blue" title ""

