#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <omp.h>

double x_ini, x_fin;
double y_ini, y_fin;

// ------------------------------------------------
// Fuentes para los casos
// ------------------------------------------------
// Caso 1: ∇²V = (x² + y²)e^{x y} (Ejemplo 1)
double source_term1(double x, double y) {
    return (x * x + y * y) * std::exp(x * y);
}

// Caso 2: ∇²V = 0 (Laplace, Ejemplo 2)
double source_term2(double x, double y) {
    return 0.0;
}

// Caso 3: ∇²V = 4 (Ejemplo 3)
double source_term3(double x, double y) {
    return 4.0;
}

// Caso 4: ∇²V = x/y + y/x (Ejemplo 4)
double source_term4(double x, double y) {
    return x / y + y / x;
}

// ------------------------------------------------
// Condiciones de frontera para los casos
// ------------------------------------------------
// Caso 1: ∇²V = (x² + y²)e^{x y}
double boundary_condition1(double x, double y) {
    const double EPS = 1e-12;
    if (std::abs(y - y_ini) < EPS) return 1.0;
    else if (std::abs(x - x_ini) < EPS) return 1.0;
    else if (std::abs(y - y_fin) < EPS) return std::exp(x);
    else if (std::abs(x - x_fin) < EPS) return std::exp(2.0 * y);
    return 0.0;
}

// Caso 2: ∇²V = 0
double boundary_condition2(double x, double y) {
    const double EPS = 1e-12;
    if (std::abs(x - x_ini) < EPS) return std::log(y*y + 1.0);
    if (std::abs(x - x_fin) < EPS) return std::log(y*y + 4.0);
    if (std::abs(y - y_ini) < EPS) return 2.0 * std::log(x);
    if (std::abs(y - y_fin) < EPS) return std::log(x*x + 4.0);
    return 0.0;
}

// Caso 3: ∇²V = 4
double boundary_condition3(double x, double y) {
    const double EPS = 1e-12;
    if (std::abs(x - x_ini) < EPS) return (1.0 - y) * (1.0 - y);
    if (std::abs(x - x_fin) < EPS) return (2.0 - y) * (2.0 - y);
    if (std::abs(y - y_ini) < EPS) return x * x;
    if (std::abs(y - y_fin) < EPS) return (x - 2.0) * (x - 2.0);
    return 0.0;
}

// Caso 4: ∇²V = x/y + y/x
double boundary_condition4(double x, double y) {
    const double EPS = 1e-12;
    if (std::abs(x - x_ini) < EPS) return y * std::log(y);
    if (std::abs(x - x_fin) < EPS) return 2.0 * y * std::log(2.0 * y);
    if (std::abs(y - y_ini) < EPS) return x * std::log(x);
    if (std::abs(y - y_fin) < EPS) return x * std::log(4.0 * x);
    return 0.0;
}

// ------------------------------------------------
// Inicialización de la grilla y condiciones de frontera
// ------------------------------------------------
void initialize_grid(int M, int N, std::vector<std::vector<double>> &V, double &h, double &k,
                     double (*boundary)(double, double)) {
    h = (x_fin - x_ini) / M;
    k = (y_fin - y_ini) / N;
    V.resize(M + 1, std::vector<double>(N + 1, 0.0));
    for (int i = 0; i <= M; ++i) {
        double x = x_ini + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_ini + j * k;
            if (i == 0 || i == M || j == 0 || j == N)
                V[i][j] = boundary(x, y);
        }
    }
}

// Precalcula la fuente
void poisson_source(int M, int N, std::vector<std::vector<double>> &F, double h, double k,
                    double (*source)(double, double)) {
    F.resize(M + 1, std::vector<double>(N + 1, 0.0));
    for (int i = 1; i < M; ++i) {
        double x = x_ini + i * h;
        for (int j = 1; j < N; ++j) {
            double y = y_ini + j * k;
            F[i][j] = source(x, y);
        }
    }
}

// Resolución de Poisson utilizando paralelización con OpenMP
void solve_poisson(std::vector<std::vector<double>> &V, const std::vector<std::vector<double>> &F,
                   int M, int N, double h, double k, double tol, int &iterations) {
    double delta = 1.0;
    iterations = 0;
    std::vector<std::vector<double>> V_old = V;

    double h2 = h * h;
    double k2 = k * k;
    double denom = 2.0 * (h2 + k2);

    while (delta > tol) {
        delta = 0.0;
        V_old = V;

        #pragma omp parallel for reduction(max:delta) collapse(2)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                double V_new = (
                    (V_old[i + 1][j] + V_old[i - 1][j]) * k2 +
                    (V_old[i][j + 1] + V_old[i][j - 1]) * h2 -
                    F[i][j] * h2 * k2
                ) / denom;

                double diff = std::abs(V_new - V[i][j]);
                if (diff > delta) delta = diff;

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

    file << "x,y,V\n";
    for (int j = 0; j <= N; ++j) {
        double y = y_ini + j * k;
        for (int i = 0; i <= M; ++i) {
            double x = x_ini + i * h;
            file << x << "," << y << "," << V[i][j] << "\n";
        }
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
        "set datafile separator \",\"\n"
        "set style data points\n"
        "set ticslevel 0\n"
        "splot '" + datafile + "' using 1:2:3 with points notitle\n"
        "exit\n";

    std::ofstream gpfile("plot_script.gp");
    gpfile << gp_script;
    gpfile.close();

    system("gnuplot plot_script.gp");
    std::remove("plot_script.gp");
}

int main() {
    int M, N, num_threads, option;

    std::cout << "Ingrese número de divisiones en x (M > 0): ";
    std::cin >> M;
    std::cout << "Ingrese número de divisiones en y (N > 0): ";
    std::cin >> N;
    std::cout << "Ingrese número de hilos (>= 1): ";
    std::cin >> num_threads;

    std::cout << "Seleccione función fuente y condiciones de frontera:\n";
    std::cout << "1) f(x,y) = (x² + y²)e^{x y} (Ejemplo 1)\n";
    std::cout << "2) f(x,y) = 0 (Laplace, Ejemplo 2)\n";
    std::cout << "3) f(x,y) = 4 (Ejemplo 3)\n";
    std::cout << "4) f(x,y) = x/y + y/x (Ejemplo 4)\n";
    std::cout << "Opción (1, 2, 3 o 4): ";
    std::cin >> option;

    if (M <= 0 || N <= 0 || num_threads < 1 || (option != 1 && option != 2 && option != 3 && option != 4)) {
        std::cerr << "Parámetros inválidos.\n";
        return 1;
    }

    double (*source_func)(double, double) = nullptr;
    double (*boundary_func)(double, double) = nullptr;

    switch (option) {
        case 1:
            x_ini = 0.0;  x_fin = 2.0;
            y_ini = 0.0;  y_fin = 1.0;
            source_func = source_term1;
            boundary_func = boundary_condition1;
            break;
        case 2:
            x_ini = 1.0;  x_fin = 2.0;
            y_ini = 0.0;  y_fin = 1.0;
            source_func = source_term2;
            boundary_func = boundary_condition2;
            break;
        case 3:
            x_ini = 1.0;  x_fin = 2.0;
            y_ini = 0.0;  y_fin = 2.0;
            source_func = source_term3;
            boundary_func = boundary_condition3;
            break;
        case 4:
            x_ini = 1.0;  x_fin = 2.0;
            y_ini = 1.0;  y_fin = 2.0;
            source_func = source_term4;
            boundary_func = boundary_condition4;
            break;
    }

    omp_set_num_threads(num_threads);
    std::cout << "Resolviendo con " << num_threads << " hilos...\n";

    double h, k;
    std::vector<std::vector<double>> V;
    std::vector<std::vector<double>> F;

    auto start_sections = std::chrono::high_resolution_clock::now();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            initialize_grid(M, N, V, h, k, boundary_func);
        }
        #pragma omp section
        {
            double dummy_h = (x_fin - x_ini) / M;
            double dummy_k = (y_fin - y_ini) / N;
            poisson_source(M, N, F, dummy_h, dummy_k, source_func);
        }
    }
    auto end_sections = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sections = end_sections - start_sections;
    std::cout << "Tiempo de ejecución de secciones: " << elapsed_sections.count() << " segundos\n";

    int iterations;
    auto start = std::chrono::high_resolution_clock::now();
    solve_poisson(V, F, M, N, h, k, 1e-6, iterations);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Convergió en " << iterations << " iteraciones\n";
    std::cout << "Tiempo de cálculo: " << elapsed.count() << " segundos\n";

    std::string option_str = (option == 1) ? "constante" : "variable";
    std::string datafile = "poisson_sections_" + std::to_string(M) + "x" + std::to_string(N) +
                           "_threads" + std::to_string(num_threads) + "_" + option_str + ".csv";
    std::string outputfile = "poisson_sections_" + std::to_string(M) + "x" + std::to_string(N) +
                             "_threads" + std::to_string(num_threads) + "_" + option_str + ".png";

    export_to_file(V, M, N, h, k, datafile);
    plot_with_gnuplot(datafile, outputfile);

    std::cout << "Gráfica generada: " << outputfile << std::endl;

    return 0;
}
