CXXFLAGS += -std=c++11 -I ..
AR = gcc-ar

.PHONY: all clean

all: libseg.a

clean:
	-rm *.o
	-rm libseg.a

libseg.a: segfeat.o fst.o scrf.o scrf_feat.o ilat.o iscrf.o util.o fscrf.o iscrf_e2e.o align.o pair_scrf.o ctc.o iscrf_e2e_ff.o segnn.o iscrf_segnn.o
	$(AR) rcs $@ $^

util.o: util.h
ctc.o: ctc.h
segnn.o: segnn.h
iscrf_segnn.o: iscrf_segnn.h
