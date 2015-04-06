CXXFLAGS += -std=c++11 -I ../../ -L ../../ebt -L ../../opt

.PHONY: all clean

all: test_scrf

clean:
	-rm *.o
	-rm test_scrf

test_scrf: test_scrf.o ../scrf.o ../lm.o ../lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lebt

