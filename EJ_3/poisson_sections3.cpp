#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <omp.h>

const double x0 = 1.0, xf = 2.0;
const double y_min = 0.0, y_max = 2.0;

// Fuente puntual, pero ahora usaremos matriz
double source_term(double x, double y) {
    return 4.0;
}

double boundary_condition(double x, double y) {
    if (std::abs(x - x0) < 1e-12) return (1.0 - y) * (1.0 - y);
    if (std::abs(x - xf) < 1e-12) return (2.0 - y) * (2.0 - y);
    if (std::abs(y - y_min) < 1e-12) return x * x;
    if (std::abs(y - y_max) < 1e-12) return (x - 2.0) * (x - 2.0);
    return 0.0;
}

// Inicializa la matriz V con condiciones de frontera
void initialize_grid(int M, int N, std::vector<std::vector<double>> &V, double &h, double &k) {
    h = (xf - x0) / M;
    k = (y_max - y_min) / N;
    V.resize(M + 1, std::vector<double>(N + 1, 0.0));

    for (int i = 0; i <= M; ++i) {
        double x = x0 + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_min + j * k;
            if (i == 0 || i == M || j == 0 || j == N)
                V[i][j] = boundary_condition(x, y);
        }
    }
}

// Nueva función que calcula la matriz fuente F para cada punto (x,y)
void compute_source_matrix(int M, int N, double h, double k, std::vector<std::vector<double>> &F) {
    F.resize(M + 1, std::vector<double>(N + 1, 0.0));

    for (int i = 0; i <= M; ++i) {
        double x = x0 + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_min + j * k;
            F[i][j] = source_term(x, y);
        }
    }
}

void solve_poisson(std::vector<std::vector<double>> &V, const std::vector<std::vector<double>> &F, int M, int N, double h, double k, double tol, int &iterations) {
    double delta = 1.0;
    iterations = 0;
    std::vector<std::vector<double>> V_old = V;

    double h2 = h * h;
    double k2 = k * k;
    double denom = 2.0 * (h2 + k2);

    while (delta > tol) {
        delta = 0.0;
        V_old = V;

        #pragma omp parallel for collapse(2) reduction(max:delta)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                double V_new = (
                    (V_old[i + 1][j] + V_old[i - 1][j]) * k2 +
                    (V_old[i][j + 1] + V_old[i][j - 1]) * h2 -
                    F[i][j] * h2 * k2
                ) / denom;

                delta = std::max(delta, std::abs(V_new - V[i][j]));
                V[i][j] = V_new;
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

    for (int j = 0; j <= N; ++j) {
        double y = y_min + j * k;
        for (int i = 0; i <= M; ++i) {
            double x = x0 + i * h;
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
        "set grid\n"
        "set hidden3d\n"
        "set style data lines\n"
        "set ticslevel 0\n"
        "splot '" + datafile + "' with lines notitle\n"
        "exit\n";

    std::ofstream gpfile("plot_script.gp");
    gpfile << gp_script;
    gpfile.close();

    system("gnuplot plot_script.gp");
    std::remove("plot_script.gp");
}

int main() {
    int M, N, num_threads;

    std::cout << "Ingrese número de divisiones en x (M > 0): ";
    std::cin >> M;
    std::cout << "Ingrese número de divisiones en y (N > 0): ";
    std::cin >> N;
    std::cout << "Ingrese número de hilos (>= 1): ";
    std::cin >> num_threads;

    if (M <= 0 || N <= 0 || num_threads < 1) {
        std::cerr << "Parámetros inválidos.\n";
        return 1;
    }

    omp_set_num_threads(num_threads);
    std::cout << "Resolviendo con " << num_threads << " hilos...\n";

    double h, k;
    std::vector<std::vector<double>> V;
    std::vector<std::vector<double>> F;

    // Paralelizamos initialize_grid y compute_source_matrix con sections
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            initialize_grid(M, N, V, h, k);
        }

        #pragma omp section
        {
            // Para calcular F, necesitamos h,k, que se inicializan en initialize_grid
            // Por eso aquí hacemos un pequeño truco: calculamos h,k antes de paralelizar
            // Para ello, los calculamos directamente:
            double h_tmp = (xf - x0) / M;
            double k_tmp = (y_max - y_min) / N;
            compute_source_matrix(M, N, h_tmp, k_tmp, F);
        }
    }

    auto end_init = std::chrono::high_resolution_clock::now();

    int iterations;
    solve_poisson(V, F, M, N, h, k, 1e-6, iterations);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_init = end_init - start;
    std::chrono::duration<double> elapsed_total = end - start;

    std::cout << "Tiempo hasta terminar initialize_grid y source: " << elapsed_init.count() << " segundos\n";
    std::cout << "Convergió en " << iterations << " iteraciones\n";

    std::cout << "Tiempo total (incluye solve_poisson): " << elapsed_total.count() << " segundos\n";

    std::string datafile = "datos_poisson.dat";
    std::string outputfile = "poisson_sections3.png";

    export_to_file(V, M, N, h, k, datafile);
    plot_with_gnuplot(datafile, outputfile);

    std::cout << "Gráfica generada: " << outputfile << std::endl;

    return 0;
}
