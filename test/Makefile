CXXFLAGS += -std=c++11 -I ../../ -L ../../ebt -L ../../opt -L ../../autodiff -L ../../la -L ../../speech

tests = test_lattice test_scrf test_fst

.PHONY: all clean

all: $(tests)

clean:
	-rm *.o
	-rm $(tests)

test_fst: test_fst.o ../lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lebt

test_lattice: test_lattice.o ../lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lebt

test_scrf: test_scrf.o ../scrf.o ../lm.o ../lattice.o ../weiran.o ../nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lautodiff -lebt -lla -lspeech

