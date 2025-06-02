#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>

const double x0 = 1.0, xf = 2.0;
const double y_min = 0.0, y_max = 2.0;

// Fuente constante: Laplaciano de V = 4 → −∇²V = −4
double source_term(double x, double y) {
    return 4.0;
}

double boundary_condition(double x, double y) {
    if (std::abs(x - x0) < 1e-12) {        // x = 1 (borde izquierdo)
        return (1.0 - y) * (1.0 - y);
    }
    if (std::abs(x - xf) < 1e-12) {        // x = 2 (borde derecho)
        return (2.0 - y) * (2.0 - y);
    }
    if (std::abs(y - y_min) < 1e-12) {     // y = 0 (borde inferior)
        return x * x;
    }
    if (std::abs(y - y_max) < 1e-12) {     // y = 2 (borde superior)
        return (x - 2.0) * (x - 2.0);
    }
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

void solve_poisson(std::vector<std::vector<double>> &V, int M, int N, double h, double k, double tol, int &iterations) {
    double delta = 1.0;
    iterations = 0;

    std::vector<std::vector<double>> V_old = V;

    while (delta > tol) {
        delta = 0.0;
        V_old = V;
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                double x = x0 + i * h;
                double y = y_min + j * k;
                double f = source_term(x, y);

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
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo para exportar datos.\n";
        return;
    }

    for (int j = 0; j <= N; ++j) {          // Recorrido por filas (y fijo) para malla correcta
        double y = y_min + j * k;
        for (int i = 0; i <= M; ++i) {      // Recorrido por columnas (x variable)
            double x = x0 + i * h;
            file << x << " " << y << " " << V[i][j] << "\n";
        }
        file << "\n";  // línea en blanco entre filas para gnuplot
    }
    file.close();
}

void plot_with_gnuplot(const std::string &datafile, const std::string &outputfile = "poisson_serial_3.png") {
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
    int M = 80, N = 80;
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
    std::string outputfile = "poisson_serial_3.png";

    export_to_file(V, M, N, h, k, datafile);
    plot_with_gnuplot(datafile, outputfile);

    std::cout << "Gráfica generada en: " << outputfile << std::endl;

    return 0;
}
