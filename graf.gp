# Script de Gnuplot para graficar solución de Poisson 2D

set title "Solución de la ecuación de Poisson 2D"
set xlabel "x"
set ylabel "y"
set zlabel "u(x,y)"
set ticslevel 0
set pm3d
set hidden3d
set view 60, 30
set palette defined ( 0 "blue", 1 "cyan", 2 "green", 3 "yellow", 4 "red" )
set colorbox
set terminal qt size 800,600
splot "solucion.dat" using 1:2:3 with pm3d notitle
