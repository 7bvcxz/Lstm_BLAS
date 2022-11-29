CC=g++
CXXFLAGS=-DMKL_ILP64 -m64

INCDIRS=-I${MKLROOT}/include
LIBDIRS=-L${MKLROOT}/lib/intel64
LIBS=-Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
OBJS= lstm.cpp
TARGET=lstm

all: $(TARGET)

clean:
	rm -f *.o
	rm -f lstm

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(INCDIRS) $(LIBDIRS) $(LIBS)


