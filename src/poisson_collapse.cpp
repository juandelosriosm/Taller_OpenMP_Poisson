#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <omp.h>

// ------------------------------------------------
// Variables globales para el dominio (se sobreescriben según opción)
// ------------------------------------------------

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
// Inicialización de la malla (grid) y condiciones de frontera
// ------------------------------------------------
void initialize_grid(int M, int N,
                     std::vector<std::vector<double>> &V,
                     double &h, double &k,
                     double (*boundary)(double, double))
{
    h = (x_fin - x_ini) / M;
    k = (y_fin - y_ini) / N;
    V.resize(M + 1, std::vector<double>(N + 1, 0.0));

    for (int i = 0; i <= M; ++i) {
        double x = x_ini + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_ini + j * k;
            if (i == 0 || i == M || j == 0 || j == N) {
                V[i][j] = boundary(x, y);
            }
        }
    }
}

// ------------------------------------------------
// Solución iterativa de ∇²V = f(x,y) por diferencias finitas (paralelo)
// ------------------------------------------------
void solve_poisson_parallel(std::vector<std::vector<double>> &V,
                            int M, int N,
                            double h, double k,
                            double tol, int &iterations,
                            double (*source)(double, double))
{
    double delta = 1.0;
    iterations = 0;
    std::vector<std::vector<double>> V_old = V;
    double h2 = h * h;
    double k2 = k * k;
    double denom = 2.0 * (h2 + k2);

    while (iterations < 15000) {
        delta = 0.0;
        V_old = V;

        // Usando collapse(2) para optimizar el anidamiento de bucles
        #pragma omp parallel for collapse(2) reduction(max:delta)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                double x = x_ini + i * h;
                double y = y_ini + j * k;
                double fval = source(x, y);

                double V_new = (
                    (V_old[i + 1][j] + V_old[i - 1][j]) * k2 +
                    (V_old[i][j + 1] + V_old[i][j - 1]) * h2 -
                    fval * h2 * k2
                ) / denom;

                double diff = std::abs(V_new - V[i][j]);
                if (diff > delta) {
                    delta = diff;
                }
                V[i][j] = V_new;
            }
        }
        ++iterations;
    }
}

// ------------------------------------------------
// Exportar datos a CSV para graficar
// ------------------------------------------------
void export_to_csv(const std::vector<std::vector<double>> &V,
                   int M, int N,
                   double h, double k,
                   const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir " << filename << " para escritura.\n";
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
    std::cout << "Archivo CSV generado: " << filename << "\n";
}

// ------------------------------------------------
// Llamada a GNUplot para generar PNG (puntos 3D)
// ------------------------------------------------
void plot_with_gnuplot(const std::string &datafile,
                       const std::string &outputfile)
{
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

// ------------------------------------------------
// Función principal
// ------------------------------------------------
int main() {
    std::cout << "=============================================\n";
    std::cout << "    Poisson/Laplace Paralelo (OpenMP)\n";
    std::cout << "=============================================\n";
    std::cout << "Este programa resuelve ∇²V = f(x,y) en 4 casos:\n\n";
    std::cout << "  1) Caso 1 (Ejemplo 3): ∇²V = 4\n";
    std::cout << "       V(1,y)=y², V(2,y)=(y-1)², V(x,0)=x², V(x,2)=(x-2)²\n";
    std::cout << "       Dominio: x∈[1,2], y∈[0,2]\n\n";

    std::cout << "  2) Caso 2 (Ejemplo 1): ∇²V = (x²+y²) e^{x y}\n";
    std::cout << "       V(0,y)=1, V(2,y)=e^{2y}, V(x,0)=1, V(x,1)=e^{x}\n";
    std::cout << "       Dominio: x∈[0,2], y∈[0,1]\n\n";

    std::cout << "  3) Caso 3 (Ejemplo 2): ∇²V = 0\n";
    std::cout << "       V(1,y)=ln(y²+1), V(2,y)=ln(y²+4), V(x,0)=2 ln(x), V(x,1)=ln(x²+4)\n";
    std::cout << "       Dominio: x∈[1,2], y∈[0,1]\n\n";

    std::cout << "  4) Caso 4 (Ejemplo 4): ∇²V = x/y + y/x\n";
    std::cout << "       V(1,y)=y ln(y), V(2,y)=2 y ln(2y), V(x,1)=x ln(x), V(x,2)=x ln(4x)\n";
    std::cout << "       Dominio: x∈[1,2], y∈[1,2]\n\n";

    std::cout << "Ingresa número de caso [1-4]: ";
    int case_selector;
    std::cin >> case_selector;

    if (case_selector < 1 || case_selector > 4) {
        std::cerr << "Opción inválida. Debe ser 1, 2, 3 o 4.\n";
        return 1;
    }

    std::cout << "Ingresa número de divisiones en x (M > 0): ";
    int M; 
    std::cin >> M;
    std::cout << "Ingresa número de divisiones en y (N > 0): ";
    int N; 
    std::cin >> N;
    std::cout << "Ingresa número de hilos OpenMP (≥ 1): ";
    int num_threads; 
    std::cin >> num_threads;

    if (M <= 0 || N <= 0 || num_threads < 1) {
        std::cerr << "Parámetros inválidos para M, N o número de hilos.\n";
        return 1;
    }

    // Fijar dominio y punteros a funciones según el caso
    double (*source_func)(double, double)       = nullptr;
    double (*boundary_func)(double, double)     = nullptr;

    switch (case_selector) {
        case 1: 
            // Caso 1: f(x,y) = (x² + y²)e^{x y}
            x_ini = 0.0;  x_fin = 2.0;
            y_ini = 0.0;  y_fin = 1.0;
            source_func = source_term1;
            boundary_func = boundary_condition1;
            break;
        case 2: 
            // Caso 2: f(x,y) = 0 (Laplace)
            x_ini = 1.0;  x_fin = 2.0;
            y_ini = 0.0;  y_fin = 1.0;
            source_func = source_term2;
            boundary_func = boundary_condition2;
            break;
        case 3: 
            // Caso 3: f(x,y) = 4
            x_ini = 1.0;  x_fin = 2.0;
            y_ini = 0.0;  y_fin = 2.0;
            source_func = source_term3;
            boundary_func = boundary_condition3;
            break;
        case 4: 
            // Caso 4: f(x,y) = x/y + y/x
            x_ini = 1.0;  x_fin = 2.0;
            y_ini = 1.0;  y_fin = 2.0;
            source_func = source_term4;
            boundary_func = boundary_condition4;
            break;
    }

    // Configurar número de hilos
    omp_set_num_threads(num_threads);
    std::cout << "Resolviendo con " << num_threads << " hilos...\n";

    // Inicializar la grilla y condición de frontera
    double h, k;
    std::vector<std::vector<double>> V; 
    initialize_grid(M, N, V, h, k, boundary_func);

    // Resolver iterativamente ∇²V = f(x,y)
    int iterations = 0;
    const double tol = 1e-6;
    auto t_start = std::chrono::high_resolution_clock::now();
    solve_poisson_parallel(V, M, N, h, k, tol, iterations, source_func);
    auto t_end   = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t_end - t_start;
    std::cout << "Convergió en " << iterations << " iteraciones.\n";
    std::cout << "Tiempo de cómputo: " << elapsed.count() << " segundos.\n";

    // Exportar a CSV y graficar con GNUplot
    std::string label; 
    switch (case_selector) {
        case 1: label = "caso1_4const";   break;
        case 2: label = "caso2_xyexpo";  break;
        case 3: label = "caso3_laplace"; break;
        case 4: label = "caso4_xyover";  break;
    }

    std::string datafile = "poisson_parallel_" + std::to_string(M) + "x" 
                         + std::to_string(N) + "_threads" 
                         + std::to_string(num_threads) + "_" + label + ".csv";

    std::string pngfile  = "poisson_parallel_" + std::to_string(M) + "x" 
                         + std::to_string(N) + "_threads" 
                         + std::to_string(num_threads) + "_" + label + ".png";

    export_to_csv(V, M, N, h, k, datafile);
    plot_with_gnuplot(datafile, pngfile);

    std::cout << "Gráfica generada en: " << pngfile << "\n";
    return 0;
}
