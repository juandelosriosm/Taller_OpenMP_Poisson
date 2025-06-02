# Makefile para varios códigos Poisson con OpenMP y optimización -O3

CXX = g++
CXXFLAGS = -fopenmp -O3
PROGS = poisson_tareas poisson_sincronizacion3 poisson_serial_3 poisson_sections3 \
        poisson_schedule3 poisson_parallel1_3 poisson_contar_iteraciones3 poisson_collapse_3

.PHONY: all clean run

all: $(PROGS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(PROGS) *.o *.png

run:
	@echo "Elige qué programa quieres compilar y ejecutar:"
	@echo ""
	@$(foreach p, $(PROGS), echo "  $(p)";)
	@read -p "Nombre del programa (sin .cpp): " prog; \
	if echo "$(PROGS)" | grep -qw $$prog; then \
		echo "Compilando $$prog.cpp..."; \
		$(CXX) $(CXXFLAGS) -o $$prog $$prog.cpp && \
		echo "Ejecutando $$prog con time..."; \
		bash -c "time ./$$prog"; \
	else \
		echo "Programa '$$prog' no encontrado."; \
	fi
