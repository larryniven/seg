CXXFLAGS += -std=c++11 -I ../

all: lm.o lattice.o scrf.o

clean:
	-rm *.o

scrf.o: fst.h lattice.h lm.h
lm.o: lm.h
lattice.o: lattice.h
