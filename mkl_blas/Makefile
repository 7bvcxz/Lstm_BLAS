CC=g++ -L /opt/intel/mkl/include
CXXFLAGS= -Wall -O3 -fPIC -std=c++12 
INCDIRS=-I/opt/intel/mkl/include
LIBDIRS=-L/opt/intel/mkl/lib/intel64
LIBS=-Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
OBJS= lstm.cpp
TARGET=lstm

all: $(TARGET)

clean:
	rm -f *.o
	rm -f lstm

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(INCDIRS) $(LIBDIRS) $(LIBS)


