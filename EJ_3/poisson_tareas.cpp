#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

const double TOL = 1e-6;

void initialize_grid(std::vector<std::vector<double>>& V, int M, int N) {
    for (int i = 0; i <= M + 1; ++i)
        for (int j = 0; j <= N + 1; ++j)
            V[i][j] = 0.0;
}

double source_term(double x, double y) {
    return std::sin(M_PI * x) * std::sin(M_PI * y);
}

void poisson_source(std::vector<std::vector<double>>& V, int M, int N, double h, double k, double x0, double y0, double x_max, double y_max) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            double x = x0 + i * h;
            double y = y0 + j * k;
            V[i][j] = 0.0;
        }
    }
}

void solve_poisson(std::vector<std::vector<double>>& V, int M, int N, double h, double k, double denom, int& iterations) {
    double h2 = h * h;
    double k2 = k * k;
    double delta;

    std::vector<std::vector<double>> V_old = V;
    double x0 = 0.0, y_min = 0.0;
    double block_size_i = M / 2;
    double block_size_j = N / 2;

    do {
        delta = 0.0;
        V_old = V;

        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int bi = 0; bi < 2; ++bi) {
                    for (int bj = 0; bj < 2; ++bj) {
                        int i0 = 1 + bi * block_size_i;
                        int j0 = 1 + bj * block_size_j;
                        int i_max = (bi == 1) ? M + 1 : i0 + block_size_i;
                        int j_max = (bj == 1) ? N + 1 : j0 + block_size_j;

                        #pragma omp task firstprivate(i0, j0, i_max, j_max) shared(V, V_old, h, k, h2, k2, denom, delta)
                        {
                            double task_delta = 0.0;

                            for (int i = i0; i < i_max; ++i) {
                                for (int j = j0; j < j_max; ++j) {
                                    double x = x0 + i * h;
                                    double y = y_min + j * k;
                                    double f = source_term(x, y);

                                    double V_new = (
                                        (V_old[i + 1][j] + V_old[i - 1][j]) * k2 +
                                        (V_old[i][j + 1] + V_old[i][j - 1]) * h2 -
                                        f * h2 * k2
                                    ) / denom;

                                    double diff = std::abs(V_new - V[i][j]);
                                    if (diff > task_delta)
                                        task_delta = diff;

                                    V[i][j] = V_new;
                                }
                            }

                            #pragma omp critical
                            {
                                if (task_delta > delta)
                                    delta = task_delta;
                            }
                        }
                    }
                }
                #pragma omp taskwait
            }
        }
        iterations++;
    } while (delta > TOL);
}

int main() {
    int M, N, iterations = 0;
    std::cout << "Ingrese número de divisiones en x (M > 0): ";
    std::cin >> M;
    std::cout << "Ingrese número de divisiones en y (N > 0): ";
    std::cin >> N;

    if (M <= 0 || N <= 0) {
        std::cerr << "Parámetros inválidos.\n";
        return 1;
    }

    double x0 = 0.0, x1 = 1.0;
    double y0 = 0.0, y1 = 1.0;
    double h = (x1 - x0) / (M + 1);
    double k = (y1 - y0) / (N + 1);
    double denom = 2.0 * (1.0 / (h * h) + 1.0 / (k * k));

    std::vector<std::vector<double>> V(M + 2, std::vector<double>(N + 2));

    double start_time = omp_get_wtime();
    initialize_grid(V, M, N);
    poisson_source(V, M, N, h, k, x0, y0, x1, y1);
    solve_poisson(V, M, N, h, k, denom, iterations);
    double end_time = omp_get_wtime();

    std::cout << "Tiempo de ejecución: " << (end_time - start_time) << " segundos\n";
    std::cout << "Iteraciones: " << iterations << "\n";

    return 0;
}
