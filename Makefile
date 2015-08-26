CXXFLAGS += -std=c++11 -I .. -L ../ebt -L ../opt -L ../speech -L ../autodiff -L ../la

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
    scrf_util.o \
    weiran.o

bin = learn learn2 learn-lat predict predict2 prune oracle-error annotate-weight predict-lat

all: $(obj) $(bin)

clean:
	-rm *.o
	-rm $(bin)

learn: learn.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

learn2: learn2.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

learn-lat: learn-lat.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict: predict.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict2: predict2.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict-lat: predict-lat.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

prune: prune.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

oracle-error: oracle-error.o lm.o lattice.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

annotate-weight: annotate-weight.o $(obj)
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
