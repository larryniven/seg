CXXFLAGS += -std=c++11 -I ../ -L ../ebt -L ../opt

bin = learn predict prune check-gold

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

learn: learn.o scrf.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lebt

predict: predict.o scrf.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lebt

prune: prune.o scrf.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lebt

check-gold: check-gold.o scrf.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lebt

scrf.o: scrf.h fst.h lattice.h lm.h
lm.o: lm.h
lattice.o: lattice.h
learn.o: fst.h scrf.h util.h
prune.o: fst.h scrf.h util.h
check-gold.o: fst.h scrf.h util.h
