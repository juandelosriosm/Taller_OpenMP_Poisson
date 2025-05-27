#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <omp.h>

const double x_ini = 0.0, x_fin = 2.0;
const double y_ini = 0.0, y_fin = 1.0;

// Fuente f(x,y)
double f(double x, double y) {
    return (x * x + y * y) * std::exp(x * y);
}

// Condiciones de frontera no homogéneas
double boundary_condition(double x, double y) {
    if (std::abs(y - y_ini) < 1e-12)       return 1.0;
    else if (std::abs(x - x_ini) < 1e-12)  return 1.0;
    else if (std::abs(y - y_fin) < 1e-12)  return std::exp(x);
    else if (std::abs(x - x_fin) < 1e-12)  return std::exp(2.0 * y);
    return 0.0;
}

// Inicialización y fuente en paralelo
void initialize_grid(int M, int N,
                     std::vector<std::vector<double>> &u,
                     std::vector<std::vector<double>> &rho,
                     double &h, double &k) {
    h = (x_fin - x_ini) / M;
    k = (y_fin - y_ini) / N;
    u.assign(M+1, std::vector<double>(N+1, 0.0));
    rho.assign(M+1, std::vector<double>(N+1, 0.0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x = x_ini + i * h;
            double y = y_ini + j * k;
            if (i==0 || i==M || j==0 || j==N)
                u[i][j] = boundary_condition(x, y);
            rho[i][j] = f(x, y);
        }
    }
}

void solve_poisson_jacobi(int M, int N,
                          std::vector<std::vector<double>> &u,
                          const std::vector<std::vector<double>> &rho,
                          double h, double k,
                          int max_iter, double tol,
                          const char* strategy) {
    double h2 = h*h, k2 = k*k;
    double denom = 2.0*(1.0/h2 + 1.0/k2);
    std::vector<std::vector<double>> u_new = u;

    double start = omp_get_wtime();

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_error = 0.0;

        #pragma omp parallel
        {
            #pragma omp for schedule(static) reduction(max:max_error) nowait
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    u_new[i][j] = ((u[i+1][j]+u[i-1][j])/h2 +
                                   (u[i][j+1]+u[i][j-1])/k2 +
                                   rho[i][j]) / denom;
                }
            }
            #pragma omp barrier
        }

        u.swap(u_new);

        for (int i = 1; i < M; ++i)
            for (int j = 1; j < N; ++j)
                max_error = std::max(max_error, std::abs(u[i][j] - u_new[i][j]));

        if (max_error < tol) {
            double end = omp_get_wtime();
            std::cout << "[" << strategy << "] Convergió en " << iter
                      << " iter con error " << max_error
                      << " en " << (end-start) << " s\n";
            return;
        }
    }

    double end = omp_get_wtime();
    std::cout << "[" << strategy << "] No convergió en "
              << max_iter << " iter en " << (end-start) << " s\n";
}

void export_to_file(const std::vector<std::vector<double>> &u,
                    double h, double k, int M, int N,
                    const std::string &filename) {
    std::ofstream file(filename);
    for (int i = 0; i <= M; ++i) {
        double x = x_ini + i*h;
        for (int j = 0; j <= N; ++j) {
            double y = y_ini + j*k;
            file << x << " " << y << " " << u[i][j] << "\n";
        }
        file << "\n";
    }
}

void create_gnuplot_script(const std::string &datafile,
                           const std::string &scriptfile) {
    std::ofstream gp(scriptfile);
    gp << "set terminal wxt size 800,600 enhanced font 'Arial,12'\n";
    gp << "set title 'Solución de Poisson'\n";
    gp << "set xlabel 'x'\nset ylabel 'y'\nset zlabel 'u(x,y)'\n";
    gp << "set pm3d at s\nset palette rgbformulae 33,13,10\n";
    gp << "splot '" << datafile << "' with pm3d\n";
    gp << "pause -1 'ENTER para salir'\n";
}

int main() {
    int M = 500, N = 500;
    double h, k;
    std::vector<std::vector<double>> u, rho;

    initialize_grid(M, N, u, rho, h, k);

    solve_poisson_jacobi(M, N, u, rho, h, k, 10000, 1e-6, "Inicialización y fuente en paralelo");

    export_to_file(u, h, k, M, N, "solucion.dat");
    create_gnuplot_script("solucion.dat", "plot.gnu");
    system("gnuplot plot.gnu");

    return 0;
}
