#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <omp.h>

const double x_ini = 0.0, x_fin = 2.0;
const double y_ini = 0.0, y_fin = 1.0;

double f(double x, double y) {
    return (x * x + y * y) * std::exp(x * y);
}

double boundary_condition(double x, double y) {
    if (std::abs(y - y_ini) < 1e-12) return 1.0;
    if (std::abs(x - x_ini) < 1e-12) return 1.0;
    if (std::abs(y - y_fin) < 1e-12) return std::exp(x);
    if (std::abs(x - x_fin) < 1e-12) return std::exp(2.0 * y);
    return 0.0;
}

void initialize_grid(int M, int N,
                     std::vector<std::vector<double>>& u,
                     std::vector<std::vector<double>>& rho,
                     double& h, double& k) {
    h = (x_fin - x_ini) / M;
    k = (y_fin - y_ini) / N;
    u.assign(M+1, std::vector<double>(N+1));
    rho.assign(M+1, std::vector<double>(N+1));

    #pragma omp parallel for collapse(2) nowait
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x = x_ini + i*h;
            double y = y_ini + j*k;
            if (i==0||i==M||j==0||j==N)
                u[i][j] = boundary_condition(x,y);
            rho[i][j] = f(x,y);
        }
    }
    // no es necesario barrier explícito aquí; el nowait evita la barrera implícita
}

void solve_poisson_jacobi(std::vector<std::vector<double>>& u,
                          const std::vector<std::vector<double>>& rho,
                          double h, double k,
                          int max_iter, double tol) {
    int M = u.size()-1, N = u[0].size()-1;
    std::vector<std::vector<double>> u_new = u;
    double h2 = h*h, k2 = k*k;
    double denom = 2*(1.0/h2 + 1.0/k2);

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_error = 0.0;

        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    u_new[i][j] = (
                        (u[i+1][j] + u[i-1][j]) / h2 +
                        (u[i][j+1] + u[i][j-1]) / k2 +
                        rho[i][j]
                    ) / denom;
                }
            }

            // sincronizar antes de intercambiar
            #pragma omp barrier

            #pragma omp single
            {
                std::cout << "Iteración " << iter << " iniciada por el hilo "
                          << omp_get_thread_num() << "\n";
            }

            #pragma omp for reduction(max:max_error)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    double err = std::abs(u_new[i][j] - u[i][j]);
                    #pragma omp critical
                    {
                        // sección artificial: actualización protegida
                        if (err > max_error) max_error = err;
                    }
                }
            }

            #pragma omp single
            {
                u.swap(u_new);
                if (max_error < tol) {
                    std::cout << "Convergió en " << iter
                              << " iteraciones, error=" << max_error << "\n";
                }
            }

            #pragma omp barrier
            if (max_error < tol) break;
        } // end parallel
    }
}

int main() {
    int M = 500, N = 500;
    double h, k;
    std::vector<std::vector<double>> u, rho;

    initialize_grid(M, N, u, rho, h, k);

    double t0 = omp_get_wtime();
    solve_poisson_jacobi(u, rho, h, k, 10000, 1e-6);
    double t1 = omp_get_wtime();

    std::cout << "Tiempo total con nowait: " << (t1 - t0) << " s\n";
    return 0;
}