#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cstdlib>

const double x_ini = 0.0, x_fin = 2.0;
const double y_ini = 0.0, y_fin = 1.0;

double f(double x, double y) {
    return (x * x + y * y) * std::exp(x * y);
}

double boundary_condition(double x, double y) {
    if (y == y_ini) return 1.0;
    if (x == x_ini) return 1.0;
    if (y == y_fin) return std::exp(x);
    if (x == x_fin) return std::exp(2 * y);
    return 0.0;
}

void initialize_grid(int M, int N, std::vector<std::vector<double>> &u, std::vector<std::vector<double>> &rho, double &h, double &k) {
    h = (x_fin - x_ini) / M;
    k = (y_fin - y_ini) / N;

    u.resize(M + 1, std::vector<double>(N + 1, 0.0));
    rho.resize(M + 1, std::vector<double>(N + 1, 0.0));

    for (int i = 0; i <= M; ++i) {
        double x = x_ini + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_ini + j * k;
            if (i == 0 || i == M || j == 0 || j == N)
                u[i][j] = boundary_condition(x, y);
            rho[i][j] = f(x, y);
        }
    }
}

void solve_poisson_jacobi(std::vector<std::vector<double>> &u, const std::vector<std::vector<double>> &rho, double h, double k, int max_iter, double tol) {
    int M = u.size() - 1;
    int N = u[0].size() - 1;

    std::vector<std::vector<double>> u_new = u;

    double h2 = h * h;
    double k2 = k * k;
    double denom = 2 * (1.0 / h2 + 1.0 / k2);

    volatile bool converged = false;

    for (int iter = 0; iter < max_iter && !converged; ++iter) {
        double max_error = 0.0;

        #pragma omp parallel
        {
            double local_error = 0.0;

            #pragma omp for collapse(2) nowait
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    u_new[i][j] = ((u[i+1][j] + u[i-1][j]) / h2 +
                                   (u[i][j+1] + u[i][j-1]) / k2 +
                                   rho[i][j]) / denom;

                    double diff = std::abs(u_new[i][j] - u[i][j]);
                    if (diff > local_error) local_error = diff;
                }
            }

            #pragma omp critical
            if (local_error > max_error) max_error = local_error;

            #pragma omp barrier

            #pragma omp for collapse(2)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    u[i][j] = u_new[i][j];
                }
            }

            #pragma omp single
            {
                if (max_error < tol) {
                    converged = true;
                    std::cout << "Convergió en " << iter << " iteraciones con error máximo " << max_error << "\n";
                }
            }
        }
    }

    if (!converged)
        std::cout << "No convergió en " << max_iter << " iteraciones\n";
}

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
    int M = 500, N = 500;
    double h, k;

    std::vector<std::vector<double>> u, rho;

    initialize_grid(M, N, u, rho, h, k);

    double t_start = omp_get_wtime();

    solve_poisson_jacobi(u, rho, h, k, 10000, 1e-6);

    double t_end = omp_get_wtime();
    std::cout << "Tiempo total: " << (t_end - t_start) << " segundos\n";

    export_to_file(u, h, k, M, N, "solucion.dat");

    std::ofstream gp("plot.gnu");
    gp << "set terminal wxt size 800,600 enhanced font 'Arial,12'\n";
    gp << "set title 'Solución de la ecuación de Poisson'\n";
    gp << "set xlabel 'x'\n";
    gp << "set ylabel 'y'\n";
    gp << "set zlabel 'u(x,y)'\n";
    gp << "set pm3d at s\n";
    gp << "set palette rgbformulae 33,13,10\n";
    gp << "splot 'solucion.dat' with pm3d\n";
    gp << "pause -1 'Presione enter para salir'\n";
    gp.close();

    system("gnuplot plot.gnu");

    return 0;
}
