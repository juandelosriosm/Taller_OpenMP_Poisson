#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>

double x_ini = 0.0, x_fin = 2.0;
double y_ini = 0.0, y_fin = 1.0;

int case_selector = 1;

// Primera función fuente: constante
double source_term1(double x, double y) {
    return 4.0;
}

// Segunda función fuente: (x² + y²)e^{xy}
double source_term2(double x, double y) {
    return (x * x + y * y) * std::exp(x * y);
}

// Condiciones de frontera para el caso 1
double boundary_condition1(double x, double y) {
    if (std::abs(x - x_ini) < 1e-12) return (1.0 - y) * (1.0 - y);
    if (std::abs(x - x_fin) < 1e-12) return (2.0 - y) * (2.0 - y);
    if (std::abs(y - y_ini) < 1e-12) return x * x;
    if (std::abs(y - y_fin) < 1e-12) return (x - 2.0) * (x - 2.0);
    return 0.0;
}

// Condiciones de frontera para el caso 2
double boundary_condition2(double x, double y) {
    if (std::abs(y - y_ini) < 1e-12) return 1.0;            // y = 0
    else if (std::abs(x - x_ini) < 1e-12) return 1.0;       // x = 0
    else if (std::abs(y - y_fin) < 1e-12) return std::exp(x); // y = 1
    else if (std::abs(x - x_fin) < 1e-12) return std::exp(2.0 * y); // x = 2
    return 0.0;
}

void initialize_grid(int M, int N, std::vector<std::vector<double>> &V, double &h, double &k) {
    h = (x_fin - x_ini) / M;
    k = (y_fin - y_ini) / N;
    V.resize(M + 1, std::vector<double>(N + 1, 0.0));

    for (int i = 0; i <= M; ++i) {
        double x = x_ini + i * h;
        for (int j = 0; j <= N; ++j) {
            double y = y_ini + j * k;
            if (i == 0 || i == M || j == 0 || j == N) {
                V[i][j] = (case_selector == 1) ? boundary_condition1(x, y) : boundary_condition2(x, y);
            }
        }
    }
}

void solve_poisson(std::vector<std::vector<double>> &V, int M, int N, double h, double k, double tol, int &iterations) {
    double delta = 1.0;
    iterations = 0;
    std::vector<std::vector<double>> V_old = V;

    while (delta > tol) {
        delta = 0.0;
        V_old = V;
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                double x = x_ini + i * h;
                double y = y_ini + j * k;
                double f = (case_selector == 1) ? source_term1(x, y) : source_term2(x, y);

                double V_new = (
                    (V_old[i + 1][j] + V_old[i - 1][j]) * (k * k) +
                    (V_old[i][j + 1] + V_old[i][j - 1]) * (h * h) -
                    f * (h * h) * (k * k)
                ) / (2.0 * (h * h + k * k));

                delta = std::max(delta, std::abs(V_new - V[i][j]));
                V[i][j] = V_new;
            }
        }
        ++iterations;
    }
}

void export_to_file(const std::vector<std::vector<double>> &V, int M, int N, double h, double k, const std::string &filename) {
    std::ofstream file(filename);
    for (int j = 0; j <= N; ++j) {
        double y = y_ini + j * k;
        for (int i = 0; i <= M; ++i) {
            double x = x_ini + i * h;
            file << x << " " << y << " " << V[i][j] << "\n";
        }
        file << "\n";
    }
    file.close();
}

void plot_with_gnuplot(const std::string &datafile, const std::string &outputfile = "poisson_plot.png") {
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
    std::cout << "Elige el caso a simular:\n";
    std::cout << "1: f(x,y) = 4 con condiciones cuadráticas\n";
    std::cout << "2: f(x,y) = (x² + y²)e^{xy} con condiciones no homogéneas\n";
    std::cin >> case_selector;

    std::cout << "Ingresa el número de divisiones M (en x): ";
    int M; std::cin >> M;

    std::cout << "Ingresa el número de divisiones N (en y): ";
    int N; std::cin >> N;

    if (case_selector == 1) {
        x_ini = 1.0; x_fin = 2.0;
        y_ini = 0.0; y_fin = 2.0;
    } else {
        x_ini = 0.0; x_fin = 2.0;
        y_ini = 0.0; y_fin = 1.0;
    }

    double h, k;
    std::vector<std::vector<double>> V;

    initialize_grid(M, N, V, h, k);

    int iterations;
    auto start = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, 1e-6, iterations);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Convergió en " << iterations << " iteraciones\n";
    std::cout << "Tiempo de cálculo: " << elapsed.count() << " segundos\n";

    std::string datafile = "datos_poisson.dat";
    std::string outputfile = (case_selector == 1) ? "poisson_case1.png" : "poisson_case2.png";

    export_to_file(V, M, N, h, k, datafile);
    plot_with_gnuplot(datafile, outputfile);

    std::cout << "Gráfica generada en: " << outputfile << std::endl;

    return 0;
}
