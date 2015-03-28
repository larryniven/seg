CXXFLAGS += -std=c++11 -I ../ -L ../ebt -L ../opt

bin = learn learn-fst

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

learn: learn.o scrf.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lebt

learn-fst: learn-fst.o scrf.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lebt

scrf.o: fst.h lattice.h lm.h
lm.o: lm.h
lattice.o: lattice.h
learn.o: fst.h scrf.h
learn-fst.o: fst.h scrf.h lm.h lattice.h
