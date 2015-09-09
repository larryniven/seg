CXXFLAGS += -std=c++11 -I .. -L ../ebt -L ../opt -L ../speech -L ../autodiff -L ../la -L ./
AR = gcc-ar

obj = lattice.o \
    lm.o \
    scrf_weight.o \
    segfeat.o \
    scrf_feat.o \
    segcost.o \
    scrf_cost.o \
    loss.o \
    nn.o \
    scrf.o \
    scrf_feat.o \
    scrf_util.o

bin = libscrf.a learn2 learn-lat predict2 prune oracle-error predict-lat

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

libscrf.a: $(obj)
	$(AR) rcs libscrf.a $(obj)

learn2: learn2.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

learn-lat: learn-lat.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict2: predict2.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict-lat: predict-lat.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

prune: prune.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

oracle-error: oracle-error.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

annotate-weight: annotate-weight.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

scrf.o: scrf.h fst.h lattice.h lm.h
lm.o: lm.h
lattice.o: lattice.h
learn.o: fst.h scrf.h util.h
learn2.o: fst.h scrf.h util.h
predict.o: fst.h scrf.h util.h
predict2.o: fst.h scrf.h util.h
prune.o: fst.h scrf.h util.h
oracle-error.o: fst.h scrf.h util.h
weiran.o: weiran.h
feat.o: feat.h scrf.h
cost.o: cost.h scrf.h
loss.o: loss.h scrf.h
nn.o: nn.h
annotate-weight.o: fst.h scrf.h util.h
predict-lat.o: fst.h scrf.h util.h
