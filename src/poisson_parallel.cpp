#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <omp.h>  // Para OpenMP

// ------------------------------------------------
// Declaración de variables globales para el dominio
// ------------------------------------------------
double x_ini = 0.0, x_fin = 2.0;
double y_ini = 0.0, y_fin = 1.0;

int case_selector = 1;  // Elegir de 1 a 4

// ------------------------------------------------
// Función fuente generalizada: ∇²V = f(x,y)
// ------------------------------------------------
double source_term(double x, double y) {
    switch (case_selector) {
        case 1:
            return (x*x + y*y) * std::exp(x * y);
        case 2:
            return 0.0;
        case 3:
            return 4.0;
        case 4:
            return x / y + y / x;
        default:
            return 0.0;
    }
}

// ------------------------------------------------
// Condiciones de frontera unificadas para cada caso
// ------------------------------------------------
static constexpr double EPS = 1e-12;

double boundary_condition(double x, double y) {
    switch (case_selector) {
        case 1:
            if (std::abs(x - x_ini) < EPS)   return 1.0;
            if (std::abs(x - x_fin) < EPS)   return std::exp(2.0 * y);
            if (std::abs(y - y_ini) < EPS)   return 1.0;
            if (std::abs(y - y_fin) < EPS)   return std::exp(x);
            return 0.0;

        case 2:
            if (std::abs(x - x_ini) < EPS)   return std::log(y*y + 1.0);
            if (std::abs(x - x_fin) < EPS)   return std::log(y*y + 4.0);
            if (std::abs(y - y_ini) < EPS)   return 2.0 * std::log(x);
            if (std::abs(y - y_fin) < EPS)   return std::log(x*x + 1.0);
            return 0.0;

        case 3:
            if (std::abs(x - x_ini) < EPS)   return (1.0 - y) * (1.0 - y);
            if (std::abs(x - x_fin) < EPS)   return (2.0 - y) * (2.0 - y);
            if (std::abs(y - y_ini) < EPS)   return x * x;
            if (std::abs(y - y_fin) < EPS)   return (x - 2.0) * (x - 2.0);
            return 0.0;

        case 4:
            if (std::abs(x - x_ini) < EPS)   return y * std::log(y);
            if (std::abs(x - x_fin) < EPS)   return 2.0 * y * std::log(2.0 * y);
            if (std::abs(y - y_ini) < EPS)   return x * std::log(x);
            if (std::abs(y - y_fin) < EPS)   return 2.0 * x * std::log(2.0 * x);
            return 0.0;

        default:
            return 0.0;
    }
}

// ------------------------------------------------
// Inicialización de la malla (grid) y condiciones de frontera
// ------------------------------------------------
void initialize_grid(int M, int N, std::vector<std::vector<double>> &V, double &h, double &k) {
    h = (x_fin - x_ini) / M;
    k = (y_fin - y_ini) / N;
    V.resize(M + 1, std::vector<double>(N + 1, 0.0));

    for (int i = 0; i <= M; ++i) {
        double x = x_ini + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_ini + j * k;
            if (i == 0 || i == M || j == 0 || j == N) {
                V[i][j] = boundary_condition(x, y);
            }
        }
    }
}

// ------------------------------------------------
// Solución iterativa de Poisson/Laplace por diferencias finitas
//   Modificado para iterar siempre 15000 veces.
//   Ahora con paralelización OpenMP y cálculo de delta.
// ------------------------------------------------
void solve_poisson(std::vector<std::vector<double>> &V,
                   int M, int N,
                   double h, double k,
                   double /*tol*/,    // ahora ignoramos la tolerancia
                   int &iterations)
{
    std::vector<std::vector<double>> V_old = V;

    for (int iter = 0; iter < 15000; ++iter) {
        V_old = V;
        double delta = 0.0;

        #pragma omp parallel for reduction(max:delta)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                double x = x_ini + i * h;
                double y = y_ini + j * k;
                double f = source_term(x, y);

                double numer = (V_old[i + 1][j] + V_old[i - 1][j]) * (k * k)
                             + (V_old[i][j + 1] + V_old[i][j - 1]) * (h * h)
                             - f * (h * h) * (k * k);
                double denom = 2.0 * (h * h + k * k);

                double V_new = numer / denom;
                double diff = std::fabs(V_new - V_old[i][j]);

                V[i][j] = V_new;
                if (diff > delta) {
                    delta = diff;
                }
            }
        }
    }

    iterations = 15000;
}

// ------------------------------------------------
// Exportar resultados a CSV para graficar
// ------------------------------------------------
std::string export_to_csv(const std::vector<std::vector<double>> &V,
                          int M, int N,
                          double h, double k,
                          int case_num)
{
    // Nombre solicitado: poisson_parallel_caso_<case_num>_<MxN>.csv
    std::string filename = "poisson_parallel_caso_" +
                           std::to_string(case_num) + "_" +
                           std::to_string(M) + "x" +
                           std::to_string(N) + ".csv";
    std::ofstream file(filename);

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
    return filename;
}

// ------------------------------------------------
// Generación de gráfica con GNUplot (3D surface o points)
// ------------------------------------------------
void plot_with_gnuplot(const std::string &csvfile,
                       const std::string &outputfile,
                       int M, int N)
{
    std::string gp_script;
    if (M > 200 && N > 200) {
        gp_script =
            "set datafile separator ','\n"
            "set terminal pngcairo size 800,600 enhanced font 'Verdana,10'\n"
            "set output '" + outputfile + "'\n"
            "set xlabel 'x'\n"
            "set ylabel 'y'\n"
            "set zlabel 'V(x,y)'\n"
            "set grid\n"
            "unset key\n"
            "set pm3d at s\n"
            "set view 60,30\n"
            "splot '" + csvfile + "' using 1:2:3 with pm3d\n";
    } else {
        gp_script =
            "set datafile separator ','\n"
            "set terminal pngcairo size 800,600 enhanced font 'Verdana,10'\n"
            "set output '" + outputfile + "'\n"
            "set xlabel 'x'\n"
            "set ylabel 'y'\n"
            "set zlabel 'V(x,y)'\n"
            "set grid\n"
            "set hidden3d\n"
            "set style data points\n"
            "set ticslevel 0\n"
            "splot '" + csvfile + "' using 1:2:3 with points notitle\n";
    }

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
    std::cout << "---------------------------------------------\n";
    std::cout << "  Simulación de Poisson/Laplace (4 ejemplos)\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "Elige el caso a simular (1-4):\n";
    std::cout << "  1: Caso 1: ∇²V = (x² + y²)e^{x y}    (V = e^{x y})\n";
    std::cout << "  2: Caso 2: ∇²V = 0   (Laplace)       (V = ln(x² + y²))\n";
    std::cout << "  3: Caso 3: ∇²V = 4                  (V = (x − y)²)\n";
    std::cout << "  4: Caso 4: ∇²V = x/y + y/x         (V = x·y·ln(x·y))\n";
    std::cout << "Ingresa el número de caso [1-4]: ";
    std::cin >> case_selector;

    if (case_selector == 1) {
        x_ini = 0.0;  x_fin = 2.0;
        y_ini = 0.0;  y_fin = 1.0;
    }
    else if (case_selector == 2) {
        x_ini = 1.0;  x_fin = 2.0;
        y_ini = 0.0;  y_fin = 1.0;
    }
    else if (case_selector == 3) {
        x_ini = 1.0;  x_fin = 2.0;
        y_ini = 0.0;  y_fin = 2.0;
    }
    else if (case_selector == 4) {
        x_ini = 1.0;  x_fin = 2.0;
        y_ini = 1.0;  y_fin = 2.0;
    }
    else {
        std::cerr << "Opción inválida. Debe ser un entero entre 1 y 4.\n";
        return 1;
    }

    std::cout << "Ingresa el número de divisiones M (en x): ";
    int M; std::cin >> M;
    std::cout << "Ingresa el número de divisiones N (en y): ";
    int N; std::cin >> N;

    std::cout << "Ingresa el número de hilos a utilizar: ";
    int num_threads;
    std::cin >> num_threads;
    omp_set_num_threads(num_threads);

    double h, k;
    std::vector<std::vector<double>> V;
    initialize_grid(M, N, V, h, k);

    int iterations = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, 1e-6, iterations);
    auto t_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t_end - t_start;
    std::cout << "Iteraciones realizadas: " << iterations << "\n";
    std::cout << "Tiempo de cálculo: " << elapsed.count() << " segundos.\n";

    // Ahora los archivos se guardan como poisson_parallel_caso_<case>_<MxN>.csv/.png
    std::string csv_file = export_to_csv(V, M, N, h, k, case_selector);
    std::string plot_file = "poisson_parallel_caso_" +
                            std::to_string(case_selector) + "_" +
                            std::to_string(M) + "x" +
                            std::to_string(N) + ".png";

    plot_with_gnuplot(csv_file, plot_file, M, N);
    std::cout << "Gráfica generada en: " << plot_file << "\n\n";

    return 0;
}
