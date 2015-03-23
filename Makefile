CXXFLAGS += -std=c++11 -I ../

all: lm.o lattice.o

lm.o: lm.h
lattice.o: lattice.h
