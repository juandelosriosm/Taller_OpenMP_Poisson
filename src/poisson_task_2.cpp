#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <omp.h>  // OpenMP

const double x0 = 1.0, xf = 2.0;
const double y_min = 0.0, y_max = 1.0;

double source_term(double x, double y) {
    return 0.0;
}

double boundary_condition(double x, double y) {
    if (fabs(x - x0) < 1e-14)
        return std::log(y * y + 1.0);
    else if (fabs(x - xf) < 1e-14)
        return std::log(y * y + 4.0);
    else if (fabs(y - y_min) < 1e-14)
        return 2.0 * std::log(x);
    else if (fabs(y - y_max) < 1e-14)
        return std::log(x * x + 4.0);
    else
        return 0.0;
}

void initialize_grid(int M, int N, std::vector<std::vector<double>> &V, double &h, double &k) {
    h = (xf - x0) / M;
    k = (y_max - y_min) / N;
    V.resize(M + 1, std::vector<double>(N + 1, 0.0));

    for (int i = 0; i <= M; ++i) {
        double x = x0 + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_min + j * k;
            if (i == 0 || i == M || j == 0 || j == N) {
                V[i][j] = boundary_condition(x, y);
            }
        }
    }
}

void solve_poisson(std::vector<std::vector<double>> &V, int M, int N, double h, double k, double tol, int &iterations, const std::string& schedule_type = "static") {
    double delta = 1.0;
    iterations = 0;

    std::vector<std::vector<double>> V_old = V;

    while (delta > tol) {
        delta = 0.0;
        V_old = V;

        #pragma omp parallel
        {
            double local_delta = 0.0;

            if (schedule_type == "static") {
                #pragma omp for reduction(max:delta) schedule(static)
                for (int i = 1; i < M; ++i) {
                    for (int j = 1; j < N; ++j) {
                        double V_new = (
                            (V_old[i + 1][j] + V_old[i - 1][j]) * (k * k) +
                            (V_old[i][j + 1] + V_old[i][j - 1]) * (h * h)
                        ) / (2.0 * (h * h + k * k));
                        delta = std::max(delta, std::abs(V_new - V[i][j]));
                        V[i][j] = V_new;
                    }
                }
            } else if (schedule_type == "dynamic") {
                #pragma omp for reduction(max:delta) schedule(dynamic)
                for (int i = 1; i < M; ++i) {
                    for (int j = 1; j < N; ++j) {
                        double V_new = (
                            (V_old[i + 1][j] + V_old[i - 1][j]) * (k * k) +
                            (V_old[i][j + 1] + V_old[i][j - 1]) * (h * h)
                        ) / (2.0 * (h * h + k * k));
                        delta = std::max(delta, std::abs(V_new - V[i][j]));
                        V[i][j] = V_new;
                    }
                }
            }
        }

        ++iterations;
    }
}

void export_to_file(const std::vector<std::vector<double>> &V, int M, int N, double h, double k, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo para exportar datos.\n";
        return;
    }

    for (int i = 0; i <= M; ++i) {
        double x = x0 + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_min + j * k;
            file << x << " " << y << " " << V[i][j] << "\n";
        }
        file << "\n";
    }
    file.close();
}

void plot_with_gnuplot(const std::string &datafile, const std::string &outputfile) {
    std::string gp_script = 
        "set terminal pngcairo size 800,600 enhanced font 'Verdana,10'\n"
        "set output '" + outputfile + "'\n"
        "set xlabel 'x'\n"
        "set ylabel 'y'\n"
        "set zlabel 'V(x,y)'\n"
        "set pm3d at s\n"
        "set style data pm3d\n"
        "set ticslevel 0\n"
        "splot '" + datafile + "' with pm3d notitle\n"
        "exit\n";

    std::ofstream gpfile("plot_script.gp");
    gpfile << gp_script;
    gpfile.close();

    system("gnuplot plot_script.gp");
    std::remove("plot_script.gp");
}

int main() {
    int M = 100, N = 100;
    double h, k;
    std::vector<std::vector<double>> V;

    initialize_grid(M, N, V, h, k);

    int iterations;
    int num_threads = omp_get_max_threads();

    for (std::string strategy : {"static", "dynamic"}) {
        std::vector<std::vector<double>> V_copy = V;
        auto start = std::chrono::high_resolution_clock::now();
        solve_poisson(V_copy, M, N, h, k, 1e-6, iterations, strategy);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Estrategia: schedule(" << strategy << ")\n";
        std::cout << "Número de threads usados: " << num_threads << "\n";
        std::cout << "Iteraciones: " << iterations << "\n";
        std::cout << "Tiempo de ejecución: " << elapsed.count() << " segundos\n\n";

        if (strategy == "static") {
            export_to_file(V_copy, M, N, h, k, "poisson_static.dat");
            plot_with_gnuplot("poisson_static.dat", "poisson_task_static_2.png");
        } else {
            export_to_file(V_copy, M, N, h, k, "poisson_dynamic.dat");
            plot_with_gnuplot("poisson_dynamic.dat", "poisson_task_dynamic_2.png");
        }
    }

    return 0;
}
