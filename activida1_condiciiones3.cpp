#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>

const double x0 = 1.0, xf = 2.0;
const double y0 = 0.0, yf = 1.0;
const double TOL = 1e-6;

// Inicializa la grilla con las condiciones de frontera del ejemplo 3
void initialize_grid(int M, int N, std::vector<std::vector<double>>& T, std::vector<std::vector<double>>& source, double& h, double& k) {
    h = (xf - x0) / M;
    k = (yf - y0) / N;

    T.resize(M + 1, std::vector<double>(N + 1, 0.0));
    source.resize(M + 1, std::vector<double>(N + 1, 4.0));  // Fuente constante: 4

    for (int j = 0; j <= N; ++j) {
        double y = y0 + j * k;
        T[0][j] = y * y;                    // V(1,y) = y^2
        T[M][j] = (y - 1) * (y - 1);        // V(2,y) = (y - 1)^2
    }
    for (int i = 0; i <= M; ++i) {
        double x = x0 + i * h;
        T[i][0] = x * x;                    // V(x,0) = x^2
        T[i][N] = (x - 2) * (x - 2);        // V(x,1) = (x - 2)^2
    }
}

// Resolver la ecuación de Poisson con paralelismo usando OpenMP y reduction
void solve_poisson(std::vector<std::vector<double>>& T, const std::vector<std::vector<double>>& source, int M, int N, double h, double k) {
    double delta = 1.0;
    int threads_used = 0;

    while (delta > TOL) {
        delta = 0.0;

        #pragma omp parallel
        {
            #pragma omp single
            {
                threads_used = omp_get_num_threads();
            }
        }

        #pragma omp parallel for reduction(max: delta) collapse(2)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                double T_new = (
                    ((T[i + 1][j] + T[i - 1][j]) * k * k) +
                    ((T[i][j + 1] + T[i][j - 1]) * h * h) -
                    (source[i][j] * h * h * k * k)) /
                    (2.0 * (h * h + k * k));

                delta = std::max(delta, std::abs(T_new - T[i][j]));
                T[i][j] = T_new;
            }
        }
    }

    std::cout << "Núcleos usados: " << threads_used << std::endl;
}

// Exporta resultados a archivo
void export_to_file(const std::vector<std::vector<double>>& T, double h, double k, int M, int N, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo para escritura.\n";
        return;
    }

    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x = x0 + i * h;
            double y = y0 + j * k;
            file << x << "\t" << y << "\t" << T[i][j] << "\n";
        }
    }

    file.close();
    std::cout << "Archivo exportado: " << filename << std::endl;
}

int main() {
    int M = 100, N = 100;
    double h, k;
    std::vector<std::vector<double>> T, source;

    initialize_grid(M, N, T, source, h, k);

    auto start = std::chrono::high_resolution_clock::now();
    solve_poisson(T, source, M, N, h, k);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Tiempo de ejecución: " << elapsed.count() << " segundos\n";

    export_to_file(T, h, k, M, N, "solucion_ejemplo3.dat");

    return 0;
}
