CXXFLAGS += -std=c++11 -I .. -L ../ebt -L ../opt -L ../speech -L ../autodiff -L ../la

bin = learn predict prune oracle-error

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

learn: learn.o scrf.o lm.o lattice.o nn.o weiran.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict: predict.o scrf.o lm.o lattice.o nn.o weiran.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

prune: prune.o scrf.o lm.o lattice.o nn.o weiran.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

oracle-error: oracle-error.o scrf.o lm.o lattice.o nn.o weiran.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

scrf.o: scrf.h fst.h lattice.h lm.h
lm.o: lm.h
lattice.o: lattice.h
learn.o: fst.h scrf.h util.h
predict.o: fst.h scrf.h util.h
prune.o: fst.h scrf.h util.h
oracle-error.o: fst.h scrf.h util.h
weiran.o: weiran.h
nn.o: nn.h
