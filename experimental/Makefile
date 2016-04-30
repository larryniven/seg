CXXFLAGS += -std=c++11 -I ../../ -L ../../speech -L ../../nn -L ../../autodiff -L ../../opt -L ../../la -L ../../ebt

bin = learn-order1 predict-order1 learn-latent-order1 learn-order2 learn-order1-e2e predict-order1-e2e
obj = segfeat.o fst.o scrf.o scrf_feat.o ilat.o iscrf.o iscrf_e2e.o pair_scrf.o align.o

.PHONY: all clean

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

learn-order1: learn-order1.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1: predict-order1.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-latent-order1: learn-latent-order1.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order2: learn-order2.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-e2e: learn-order1-e2e.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-e2e: predict-order1-e2e.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

