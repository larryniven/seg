CXXFLAGS += -std=c++11 -I ../../ -L ../../ebt -L ../../opt -L ../../autodiff -L ../../la -L ../../speech

.PHONY: all clean

all: test_lattice test_scrf

clean:
	-rm *.o
	-rm test_lattice test_scrf

test_lattice: test_lattice.o ../lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lebt

test_scrf: test_scrf.o ../scrf.o ../lm.o ../lattice.o ../weiran.o ../nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lautodiff -lebt -lla -lspeech

