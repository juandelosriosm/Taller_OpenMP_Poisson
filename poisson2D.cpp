#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

const double x_ini = 0.0, x_fin = 2.0;
const double y_ini = 0.0, y_fin = 1.0;

// Función fuente f(x, y)
double f(double x, double y) {
    return (x * x + y * y) * std::exp(x * y);
}

// Condiciones de frontera no homogéneas
double boundary_condition(double x, double y) {
    if (std::abs(y - y_ini) < 1e-12) {       // y = 0
        return 1.0;
    } else if (std::abs(x - x_ini) < 1e-12) { // x = 0
        return 1.0;
    } else if (std::abs(y - y_fin) < 1e-12) { // y = 1
        return std::exp(x);
    } else if (std::abs(x - x_fin) < 1e-12) { // x = 2
        return std::exp(2.0 * y);
    }
    return 0.0; // Para seguridad
}

// Inicializar la malla con condiciones de frontera y fuente
void initialize_grid(int M, int N, std::vector<std::vector<double>> &u, std::vector<std::vector<double>> &rho, double &h, double &k) {
    h = (x_fin - x_ini) / M;
    k = (y_fin - y_ini) / N;

    u.resize(M + 1, std::vector<double>(N + 1, 0.0));
    rho.resize(M + 1, std::vector<double>(N + 1, 0.0));

    for (int i = 0; i <= M; ++i) {
        double x = x_ini + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_ini + j * k;

            // Condiciones de frontera no homogéneas
            if (i == 0 || i == M || j == 0 || j == N) {
                u[i][j] = boundary_condition(x, y);
            }

            // Función fuente
            rho[i][j] = f(x, y);
        }
    }
}

// Método iterativo de relajación de Jacobi
void solve_poisson_jacobi(std::vector<std::vector<double>> &u, const std::vector<std::vector<double>> &rho, double h, double k, int max_iter, double tol) {
    int M = u.size() - 1;
    int N = u[0].size() - 1;

    std::vector<std::vector<double>> u_new = u;

    double h2 = h * h;
    double k2 = k * k;
    double denom = 2.0 * (1.0 / h2 + 1.0 / k2);

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_error = 0.0;

        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                // Actualizamos solo puntos interiores
                u_new[i][j] = ((u[i+1][j] + u[i-1][j]) / h2 +
                               (u[i][j+1] + u[i][j-1]) / k2 +
                               rho[i][j]) / denom;

                max_error = std::max(max_error, std::abs(u_new[i][j] - u[i][j]));
            }
        }

        u = u_new;

        if (max_error < tol) {
            std::cout << "Convergió en " << iter << " iteraciones con error máximo " << max_error << "\n";
            return;
        }
    }

    std::cout << "No convergió en " << max_iter << " iteraciones\n";
}

// Exportar la solución a un archivo para graficar
void export_to_file(const std::vector<std::vector<double>> &u, double h, double k, int M, int N, const std::string &filename) {
    std::ofstream file(filename);
    for (int i = 0; i <= M; ++i) {
        double x = x_ini + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_ini + j * k;
            file << x << " " << y << " " << u[i][j] << "\n";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    int M = 500;  // número de divisiones en x
    int N = 500;   // número de divisiones en y
    double h, k;

    std::vector<std::vector<double>> u, rho;

    initialize_grid(M, N, u, rho, h, k);
    solve_poisson_jacobi(u, rho, h, k, 10000, 1e-6);
    export_to_file(u, h, k, M, N, "solucion.dat");

    return 0;
}
