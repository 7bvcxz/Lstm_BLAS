CC=g++
CXXFLAGS= -Wall -O3 -fPIC -std=c++11 
INCDIRS=-I${DNNLROOT}/include
LIBDIRS=-L${DNNLROOT}/lib
LIBS=-Wl,--no-as-needed -ldnnl
OBJS= lstm.cpp
TARGET=lstm

all: $(TARGET)

clean:
	rm -f *.o
	rm -f lstm

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(INCDIRS) $(LIBDIRS) $(LIBS)


