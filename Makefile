CXXFLAGS += -std=c++11 -I .. -L ../ebt -L ../opt -L ../speech -L ../autodiff -L ../la

obj = lattice.o lm.o feat.o cost.o loss.o nn.o scrf.o make_feature.o weiran.o

bin = learn learn2 predict predict2 prune oracle-error

all: $(obj) $(bin)

clean:
	-rm *.o
	-rm $(bin)

learn: learn.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

learn2: learn2.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict: predict.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

predict2: predict2.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

prune: prune.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lspeech -lla -lebt

oracle-error: oracle-error.o lm.o lattice.o
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
make_feature.o: make_feature.h weiran.h nn.h feat.h scrf.h
nn.o: nn.h
