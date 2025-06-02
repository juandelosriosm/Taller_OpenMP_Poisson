set terminal pngcairo size 800,600 enhanced font 'Arial,10'
set output 'solucion.png'
set title 'Solución de la ecuación de Poisson 2D'
set xlabel 'x'
set ylabel 'y'
set pm3d
set view map
set palette rgbformulae 22,13,-31
splot 'solucion.dat' using 1:2:3 with pm3d notitle
