CXX = g++
CXXFLAGS_COMMON = -O3
CXXFLAGS_OMP = -fopenmp -O3

SRCDIR = src
BINDIR = bin

PROGS = poisson_serial poisson_parallel poisson_collapse poisson_sections

.PHONY: all clean run dirs

all: dirs $(addprefix $(BINDIR)/, $(PROGS))

dirs:
	mkdir -p $(BINDIR)

$(BINDIR)/poisson_serial: $(SRCDIR)/poisson_serial.cpp
	$(CXX) $(CXXFLAGS_COMMON) -o $@ $<

$(BINDIR)/poisson_parallel: $(SRCDIR)/poisson_parallel.cpp
	$(CXX) $(CXXFLAGS_OMP) -o $@ $<

$(BINDIR)/poisson_collapse: $(SRCDIR)/poisson_collapse.cpp
	$(CXX) $(CXXFLAGS_OMP) -o $@ $<

$(BINDIR)/poisson_sections: $(SRCDIR)/poisson_sections.cpp
	$(CXX) $(CXXFLAGS_OMP) -o $@ $<

clean:
	rm -f $(BINDIR)/* *.o *.png *.dat *.csv

run: all
	@echo "Programas disponibles para compilar y ejecutar:"
	@$(foreach p, $(PROGS), echo "  $(p)";)
	@read -p "Nombre del programa (sin .cpp): " prog; \
	if echo "$(PROGS)" | grep -qw "$$prog"; then \
		echo "Ejecutando $(BINDIR)/$$prog con time..."; \
		time ./$(BINDIR)/$$prog; \
		if [ -x "./organizar.sh" ]; then \
			echo "Ejecutando organizar.sh..."; \
			./organizar.sh; \
		else \
			echo "Script organizar.sh no encontrado o sin permisos de ejecución."; \
		fi; \
	else \
		echo "Programa '$$prog' no encontrado."; \
	fi
