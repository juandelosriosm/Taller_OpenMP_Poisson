#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <string>

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
        //---------------------------------------------
        // Ejemplo 1: ∇²V = (x² + y²) e^{x y}
        // Dominio: x ∈ [0,2], y ∈ [0,1]
        // Solución analítica: V(x,y) = e^{x y}
        //---------------------------------------------
        case 1:
            return (x*x + y*y) * std::exp(x * y);

        //---------------------------------------------
        // Ejemplo 2: ∇²V = 0  (Ecuación de Laplace)
        // Dominio: x ∈ [1,2], y ∈ [0,1]
        // Solución analítica: V(x,y) = ln( x² + y² )
        //---------------------------------------------
        case 2:
            return 0.0;

        //---------------------------------------------
        // Ejemplo 3: ∇²V = 4
        // Dominio: x ∈ [1,2], y ∈ [0,2]
        // Solución analítica: V(x,y) = (x - 2)²  [¡NO!—ver más abajo]
        //    (en realidad la solución dada en la tabla es V(x,y) = (x - 2)²,
        //     pero verifica que ∇²((x-2)²) = 2 + 0 = 2, no 4. 
        //     Para que ∇²V = 4, la función candidata es V = x² + y² 
        //     (porque ∇²(x²+y²) = 2 + 2 = 4). 
        //     Sin embargo, para ajustar las fronteras EXACTAS dadas en la tabla 
        //     (V(1,y)=y², V(2,y)=(y-1)², V(x,0)=x², V(x,2)=(x-2)²), 
        //     basta tomar V(x,y) = (x-2)² + y². 
        //     En ese caso ∇²V = 2 + 2 = 4, y comprueba las fronteras. )
        //---------------------------------------------
        case 3:
            return 4.0;

        //---------------------------------------------
        // Ejemplo 4: ∇²V = x/y + y/x
        // Dominio: x ∈ [1,2], y ∈ [1,2]
        // Solución analítica: V(x,y) = x y ln(x y)
        //---------------------------------------------
        case 4:
            return x / y + y / x;

        default:
            return 0.0;
    }
}

// ------------------------------------------------
// Condiciones de frontera unificadas para cada caso
// ------------------------------------------------
// Usaremos un pequeño eps para comparar dobles:
static constexpr double EPS = 1e-12;

double boundary_condition(double x, double y) {
    switch (case_selector) {
        //=============================================
        // Ejemplo 1:  ∇²V = (x² + y²)e^{xy}
        //  V(0,y) = 1
        //  V(2,y) = e^{2y}
        //  V(x,0) = 1
        //  V(x,1) = e^{x}
        //  Dominio: x∈[0,2], y∈[0,1]
        //=============================================
        case 1:
            if (std::abs(x - x_ini) < EPS) {          // x = 0
                return 1.0; 
            }
            if (std::abs(x - x_fin) < EPS) {          // x = 2
                return std::exp(2.0 * y);
            }
            if (std::abs(y - y_ini) < EPS) {          // y = 0
                return 1.0;
            }
            if (std::abs(y - y_fin) < EPS) {          // y = 1
                return std::exp(x);
            }
            return 0.0;  // Interior (no se usa)

        //=============================================
        // Ejemplo 2:  ∇²V = 0   (Ecuación de Laplace)
        //  V(1,y) = ln(y² + 1)
        //  V(2,y) = ln(y² + 4)
        //  V(x,0) = 2 ln(x)
        //  V(x,1) = ln(x² + 4)
        //  Dominio: x∈[1,2], y∈[0,1]
        //=============================================
        case 2:
            if (std::abs(x - x_ini) < EPS) {          // x = 1
                return std::log(y*y + 1.0);
            }
            if (std::abs(x - x_fin) < EPS) {          // x = 2
                return std::log(y*y + 4.0);
            }
            if (std::abs(y - y_ini) < EPS) {          // y = 0
                return 2.0 * std::log(x);
            }
            if (std::abs(y - y_fin) < EPS) {          // y = 1
                return std::log(x*x + 4.0);
            }
            return 0.0;

        //=============================================
        // Ejemplo 3:  ∇²V = 4
        //  V(1,y) = y²
        //  V(2,y) = (y-1)²
        //  V(x,0) = x²
        //  V(x,2) = (x-2)²
        //  Dominio: x∈[1,2], y∈[0,2]
        //=============================================
        case 3:
            if (std::abs(x - x_ini) < EPS) {          // x = 1
                return y*y;
            }
            if (std::abs(x - x_fin) < EPS) {          // x = 2
                return (y - 1.0) * (y - 1.0);
            }
            if (std::abs(y - y_ini) < EPS) {          // y = 0
                return x*x;
            }
            if (std::abs(y - y_fin) < EPS) {          // y = 2
                return (x - 2.0) * (x - 2.0);
            }
            return 0.0;

        //=============================================
        // Ejemplo 4:  ∇²V = x/y + y/x
        //  V(1,y) = y ln(y)
        //  V(2,y) = 2 y ln(2 y)
        //  V(x,1) = x ln(x)
        //  V(x,2) = x ln(4 x)
        //  Dominio: x∈[1,2], y∈[1,2]
        //=============================================
        case 4:
            if (std::abs(x - x_ini) < EPS) {          // x = 1
                return y * std::log(y);
            }
            if (std::abs(x - x_fin) < EPS) {          // x = 2
                return 2.0 * y * std::log(2.0 * y);
            }
            if (std::abs(y - y_ini) < EPS) {          // y = 1
                return x * std::log(x);
            }
            if (std::abs(y - y_fin) < EPS) {          // y = 2
                return x * std::log(4.0 * x);
            }
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
            // Si está en la frontera (i==0, i==M, j==0, j==N), le asignamos boundary_condition:
            if (i == 0 || i == M || j == 0 || j == N) {
                V[i][j] = boundary_condition(x, y);
            }
        }
    }
}

// ------------------------------------------------
// Solución iterativa de Poisson/Laplace por diferencias finitas
// ------------------------------------------------
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
                double f = source_term(x, y);

                // Esquema de cinco puntos (5-point stencil) para ∇²V = f
                double numer = (V_old[i + 1][j] + V_old[i - 1][j]) * (k * k)
                             + (V_old[i][j + 1] + V_old[i][j - 1]) * (h * h)
                             - f * (h * h) * (k * k);
                double denom = 2.0 * (h * h + k * k);

                double V_new = numer / denom;
                delta = std::max(delta, std::abs(V_new - V[i][j]));
                V[i][j] = V_new;
            }
        }
        ++iterations;
    }
}

// ------------------------------------------------
// Exportar resultados a CSV para graficar
// ------------------------------------------------
std::string export_to_csv(const std::vector<std::vector<double>> &V, int M, int N, double h, double k, int case_num) {
    std::string filename = "poisson_" + std::to_string(M) + "x" + std::to_string(N) + "_case" + std::to_string(case_num) + ".csv";
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
    std::cout << "Archivo CSV generado: " << filename << std::endl;
    return filename;
}

// ------------------------------------------------
// Generación de gráfica con GNUplot (3D surface o points)
// ------------------------------------------------
void plot_with_gnuplot(const std::string &csvfile, const std::string &outputfile, int M, int N) {
    std::string gp_script;
    if (M > 200 && N > 200) {
        // Muchos puntos: usar pm3d para superficie
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
        // Pocos puntos: usar with points
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
    std::cout << "  1: Ejemplo 1: ∇²V = (x²+y²)e^{xy}  (V=e^{x y})\n";
    std::cout << "  2: Ejemplo 2: ∇²V = 0  (Laplace)  (V=ln(x²+y²))\n";
    std::cout << "  3: Ejemplo 3: ∇²V = 4               (V = (x-2)² + y²)\n";
    std::cout << "  4: Ejemplo 4: ∇²V = x/y + y/x       (V = x y ln(x y))\n";
    std::cout << "Ingresa el número de caso [1-4]: ";
    std::cin >> case_selector;

    // ----------------------------
    // Definir dominio (x_ini, x_fin, y_ini, y_fin) según el caso
    // ----------------------------
    if (case_selector == 1) {
        // Ejemplo 1: x ∈ [0,2], y ∈ [0,1]
        x_ini = 0.0;  x_fin = 2.0;
        y_ini = 0.0;  y_fin = 1.0;
    }
    else if (case_selector == 2) {
        // Ejemplo 2: x ∈ [1,2], y ∈ [0,1]
        x_ini = 1.0;  x_fin = 2.0;
        y_ini = 0.0;  y_fin = 1.0;
    }
    else if (case_selector == 3) {
        // Ejemplo 3: x ∈ [1,2], y ∈ [0,2]
        x_ini = 1.0;  x_fin = 2.0;
        y_ini = 0.0;  y_fin = 2.0;
    }
    else if (case_selector == 4) {
        // Ejemplo 4: x ∈ [1,2], y ∈ [1,2]
        x_ini = 1.0;  x_fin = 2.0;
        y_ini = 1.0;  y_fin = 2.0;
    }
    else {
        std::cerr << "Opción inválida. Debe ser un entero entre 1 y 4.\n";
        return 1;
    }

    // ----------------------------
    // Pedir número de divisiones
    // ----------------------------
    std::cout << "Ingresa el número de divisiones M (en x): ";
    int M; 
    std::cin >> M;
    std::cout << "Ingresa el número de divisiones N (en y): ";
    int N; 
    std::cin >> N;

    // ----------------------------
    // Preparar la malla y condiciones de frontera
    // ----------------------------
    double h, k;
    std::vector<std::vector<double>> V;
    initialize_grid(M, N, V, h, k);

    // ----------------------------
    // Resolver iterativamente Poisson/Laplace
    // ----------------------------
    int iterations = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, 1e-6, iterations);
    auto t_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t_end - t_start;
    std::cout << "Convergió en " << iterations << " iteraciones.\n";
    std::cout << "Tiempo de cálculo: " << elapsed.count() << " segundos.\n";

    // ----------------------------
    // Exportar resultados a CSV y graficar con gnuplot
    // ----------------------------
    std::string csv_file = export_to_csv(V, M, N, h, k, case_selector);
    std::string plot_file = "poisson_" + std::to_string(M) + "x" + std::to_string(N) 
                            + "_case" + std::to_string(case_selector) + ".png";

    plot_with_gnuplot(csv_file, plot_file, M, N);
    std::cout << "Gráfica generada en: " << plot_file << std::endl << std::endl;

    return 0;
}

