#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

// Dominio
const double x_ini = 0.0, x_fin = 2.0;
const double y_ini = 0.0, y_fin = 1.0;

// Solución exacta
double exact_solution(double x, double y) {
    return (x - y) * (x - y);
}

// Generar archivo CSV
void write_exact_solution_csv(const std::string& filename, int Nx, int Ny) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo " << filename << " para escribir.\n";
        return;
    }

    double dx = (x_fin - x_ini) / (Nx - 1);
    double dy = (y_fin - y_ini) / (Ny - 1);

    file << "x,y,V_teorica\n";
    for (int i = 0; i < Nx; ++i) {
        double x = x_ini + i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = y_ini + j * dy;
            double V = exact_solution(x, y);
            file << std::setprecision(8) << x << "," << y << "," << V << "\n";
        }
    }

    file.close();
    std::cout << "Archivo '" << filename << "' generado con éxito.\n";
}

int main() {
    int Nx, Ny;
    std::cout << "Ingrese el número de puntos en x (Nx): ";
    std::cin >> Nx;
    std::cout << "Ingrese el número de puntos en y (Ny): ";
    std::cin >> Ny;

    if (Nx < 2 || Ny < 2) {
        std::cerr << "La malla debe tener al menos 2x2 puntos.\n";
        return 1;
    }

    write_exact_solution_csv("solucion_teorica.csv", Nx, Ny);

    return 0;
}
