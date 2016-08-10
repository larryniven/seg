CXXFLAGS += -std=c++11 -I ../ -L ../speech -L ../nn -L ../autodiff -L ../opt -L ../la -L ../ebt

bin = \
    learn-order1 \
    predict-order1 \
    prune-order1 \
    learn-latent-order1 \
    learn-latent-order1-e2e \
    learn-order1-e2e \
    predict-order1-e2e \
    learn-order1-segnn \
    predict-order1-segnn \
    learn-order1-e2e-ff \
    learn-latent-order1-e2e-ff \
    predict-order1-e2e-ff \
    learn-fw-order1 \
    fw-duality-gap \
    oracle-error \
    oracle-cost \
    learn-ctc \
    predict-ctc \
    learn-order1-full \
    predict-order1-full \
    loss-order1-full \
    learn-order1-lat \
    forced-align-order1-full

obj = segfeat.o fst.o scrf.o scrf_feat.o ilat.o iscrf.o util.o

.PHONY: all clean

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

learn-order1: learn-order1.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1: predict-order1.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

prune-order1: prune-order1.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-latent-order1: learn-latent-order1.o align.o pair_scrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-fw-order1: learn-fw-order1.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

fw-duality-gap: fw-duality-gap.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order2: learn-order2.o $(obj) 
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-e2e: learn-order1-e2e.o iscrf_e2e.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-latent-order1-e2e: learn-latent-order1-e2e.o iscrf_e2e.o align.o pair_scrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-e2e: predict-order1-e2e.o iscrf_e2e.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-segnn: learn-order1-segnn.o segnn.o iscrf_segnn.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-segnn: predict-order1-segnn.o segnn.o iscrf_segnn.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-e2e-ff: learn-order1-e2e-ff.o iscrf_e2e_ff.o iscrf_e2e.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-latent-order1-e2e-ff: learn-latent-order1-e2e-ff.o iscrf_e2e_ff.o iscrf_e2e.o align.o pair_scrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-e2e-ff: predict-order1-e2e-ff.o iscrf_e2e_ff.o iscrf_e2e.o align.o pair_scrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-latent-order1-e2e-lstm2d: learn-latent-order1-e2e-lstm2d.o iscrf_e2e_lstm2d.o iscrf_e2e.o align.o pair_scrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-e2e-lstm2d: predict-order1-e2e-lstm2d.o iscrf_e2e_lstm2d.o iscrf_e2e.o align.o pair_scrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

oracle-error: oracle-error.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

oracle-cost: oracle-cost.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-ctc: learn-ctc.o ctc.o fst.o ilat.o util.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-ctc: predict-ctc.o ctc.o fst.o ilat.o util.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-e2e-mll: learn-order1-e2e-mll.o pair_scrf.o iscrf_e2e.o align.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-full: learn-order1-full.o fscrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-full: predict-order1-full.o fscrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

loss-order1-full: loss-order1-full.o fscrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

forced-align-order1-full: forced-align-order1-full.o fscrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-lat: learn-order1-lat.o fscrf.o $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

util.o: util.h
ctc.o: ctc.h
segnn.o: segnn.h
iscrf_segnn.o: iscrf_segnn.h
