CXXFLAGS += -std=c++11 -I .. -L ../ebt -L ../opt -L ../speech -L ../la -L ../autodiff -L ./
AR = gcc-ar

obj = fst.o \
    ilat.o \
    lattice.o \
    lm.o \
    scrf_weight.o \
    scrf_feat.o \
    scrf_feat_util.o \
    segfeat.o \
    segcost.o \
    scrf_cost.o \
    loss.o \
    scrf.o \
    iscrf.o \
    scrf_feat.o \
    scrf_util.o \
    make_feat.o \
    nn_feat.o

bin = libscrf.a \
    learn-exp \
    learn \
    learn-first \
    predict \
    predict-first \
    learn-lat \
    predict-lat \
    prune \
    prune-first \
    beam-prune \
    beam-prune-first \
    vertex-prune \
    vertex-prune-first \
    oracle-error \
    forced-align \
    lat-cost

.PHONY: all gpu clean header

all: $(bin)

gpu: learn-e2e predict-e2e

clean:
	-rm *.o *.gch
	-rm $(bin)

header: fst.h.gch

libscrf.a: $(obj)
	$(AR) rcs libscrf.a $(obj)

learn-exp: learn-exp.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

learn: learn.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

learn-first: learn-first.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

predict: predict.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

predict-first: predict-first.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

learn-lat: learn-lat.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

predict-lat: predict-lat.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

learn-e2e.o: learn-e2e.cu
	nvcc $(CXXFLAGS) -c learn-e2e.cu

e2e-util.o: e2e-util.cu
	nvcc $(CXXFLAGS) -c e2e-util.cu

learn-e2e: learn-e2e.o e2e-util.o libscrf.a
	$(CXX) $(CXXFLAGS) -L /opt/cuda/lib64 -o $@ $^ -lnngpu -lautodiffgpu -loptgpu -lspeech -llagpu -lebt -lcblas -lcublas -lcudart

predict-e2e.o: predict-e2e.cu
	nvcc $(CXXFLAGS) -c predict-e2e.cu

predict-e2e: predict-e2e.o e2e-util.o libscrf.a
	nvcc $(CXXFLAGS) -L /opt/cuda/lib64 -o $@ $^ -lnngpu -lautodiffgpu -loptgpu -lspeech -llagpu -lebt -lcblas -lcublas -lcudart

prune: prune.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

prune-first: prune-first.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

beam-prune: beam-prune.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

beam-prune-first: beam-prune-first.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

vertex-prune: vertex-prune.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

vertex-prune-first: vertex-prune-first.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

oracle-error: oracle-error.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lspeech -lla -lebt -lcblas

forced-align: forced-align.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

lat-cost: lat-cost.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

fst.o: fst.h
ilat.o: ilat.h
scrf.o: scrf.h fst.h lattice.h lm.h
lm.o: lm.h
lattice.o: lattice.h
weiran.o: weiran.h
feat.o: feat.h scrf.h
cost.o: cost.h scrf.h
loss.o: loss.h scrf.h
nn.o: nn.h
scrf_weight.o: scrf_weight.h scrf_feat.h scrf.h
scrf_feat.o: scrf_feat.h scrf.h
make_feat.o: make_feat.h scrf_feat.h
scrf_cost.o: scrf_cost.h
segcost.o: segcost.h
segfeat.o: segfeat.h
iscrf.o: iscrf.h scrf.h fst.h
pair_scrf.o: pair_scrf.h

learn.o: fst.h scrf.h util.h
learn-first.o: fst.h scrf.h util.h
predict.o: fst.h scrf.h util.h
learn-lat.o: fst.h scrf.h util.h
predict-lat.o: fst.h scrf.h util.h
learn-e2e.o: fst.h scrf.h util.h
predict-e2e.o: fst.h scrf.h util.h
prune.o: fst.h scrf.h util.h
prune-first.o: fst.h scrf.h util.h
beam-prune.o: fst.h scrf.h util.h
beam-prune-first.o: fst.h scrf.h util.h
vertex-prune.o: fst.h scrf.h util.h
oracle-error.o: fst.h scrf.h util.h
forced-align.o: fst.h scrf.h util.h
lat-cost.o: fst.h scrf.h util.h

fst.h.gch: fst.h
	$(CXX) $(CXXFLAGS) -c -o $@ $^
