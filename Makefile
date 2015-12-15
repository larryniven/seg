CXXFLAGS += -std=c++11 -I .. -L ../ebt -L ../opt -L ../speech -L ../autodiff -L ../la -L ./ -lcblas
AR = gcc-ar

obj = lattice.o \
    lm.o \
    scrf_weight.o \
    segfeat.o \
    scrf_feat.o \
    segcost.o \
    scrf_cost.o \
    loss.o \
    scrf.o \
    scrf_feat.o \
    scrf_util.o \
    make_feat.o

bin = libscrf.a learn learn-lat predict prune oracle-error predict-lat forced-align

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

libscrf.a: $(obj)
	$(AR) rcs libscrf.a $(obj)

learn: learn.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

learn-lat: learn-lat.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict: predict.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict-lat: predict-lat.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

forced-align: forced-align.o libscrf.a
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
predict.o: fst.h scrf.h util.h
prune.o: fst.h scrf.h util.h
oracle-error.o: fst.h scrf.h util.h
weiran.o: weiran.h
feat.o: feat.h scrf.h
cost.o: cost.h scrf.h
loss.o: loss.h scrf.h
nn.o: nn.h
annotate-weight.o: fst.h scrf.h util.h
predict-lat.o: fst.h scrf.h util.h
forced-align.o: fst.h scrf.h util.h

scrf_weight.o: scrf_weight.h scrf_feat.h scrf.h
scrf_feat.o: scrf_feat.h scrf.h
make_feat.o: make_feat.h scrf_feat.h
