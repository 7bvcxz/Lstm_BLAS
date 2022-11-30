CC=g++
CXXFLAGS= -Wall -O3 -fPIC -std=c++11
INCDIRS=-I/opt/OpenBLAS/include
LIBDIRS=-L/opt/OpenBLAS/lib
#LIBS=-Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
LIBS=-Wl,--no-as-needed -lopenblas
OBJS= lstm.cpp
TARGET=lstm

all: $(TARGET)

init:
	export OPENBLAS_NUM_THREADS=4
	export OMP_NUM_THREADS=4

clean:
	rm -f *.o
	rm -f lstm
	rm -f bn
lstm:
	$(CC) -o lstm lstm.cpp $(INCDIRS) $(LIBDIRS) $(LIBS)

bn1d:
	$(CC) -o bn1d bn1d.cpp $(INCDIRS) $(LIBDIRS) $(LIBS)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(INCDIRS) $(LIBDIRS) $(LIBS)


