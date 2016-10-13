CXXFLAGS += -std=c++11 -I ..
AR = gcc-ar

.PHONY: all clean

all: libseg.a

clean:
	-rm *.o
	-rm libseg.a

libseg.a: segfeat.o fst.o scrf.o scrf_feat.o ilat.o util.o fscrf.o ctc.o
	$(AR) rcs $@ $^

util.o: util.h
ctc.o: ctc.h
