CXXFLAGS += -std=c++14 -I ..
AR = gcc-ar

.PHONY: all clean

all: libseg.a

clean:
	-rm *.o
	-rm libseg.a

libseg.a: lat.o seg.o loss.o seg-weight.o seg-util.o ctc.o
	$(AR) rcs $@ $^

util.o: util.h
ctc.o: ctc.h

loss.o: loss.h loss-util.h loss-util-impl.h
