CXXFLAGS += -std=c++11 -I .. -L ../ebt -L ../opt -L ../speech -L ../la -L ../autodiff -L ../nn -L ./
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

bin = libscrf.a \
    learn \
    predict \
    learn-lat \
    predict-lat \
    prune \
    beam-prune \
    vertex-prune \
    oracle-error \
    forced-align

all: $(bin)

gpu: learn-e2e predict-e2e

clean:
	-rm *.o
	-rm $(bin)

libscrf.a: $(obj)
	$(AR) rcs libscrf.a $(obj)

learn: learn.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

predict: predict.o libscrf.a
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

beam-prune: beam-prune.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

vertex-prune: vertex-prune.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

oracle-error: oracle-error.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lopt -lspeech -lla -lebt -lcblas

forced-align: forced-align.o libscrf.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt -lcblas

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

learn.o: fst.h scrf.h util.h
predict.o: fst.h scrf.h util.h
learn-lat.o: fst.h scrf.h util.h
predict-lat.o: fst.h scrf.h util.h
learn-e2e.o: fst.h scrf.h util.h
predict-e2e.o: fst.h scrf.h util.h
prune.o: fst.h scrf.h util.h
beam-prune.o: fst.h scrf.h util.h
vertex-prune.o: fst.h scrf.h util.h
oracle-error.o: fst.h scrf.h util.h
forced-align.o: fst.h scrf.h util.h
