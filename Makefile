CC = gcc-14
CFLAGS = -fopenmp -Wall -Wextra
# Include directories
INCLUDES = -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include
# Library directories and libraries
LIBDIRS = -L/usr/local/opt/lapack/lib -L/usr/local/opt/openblas/lib
LIBS = -llapack -llapacke -lopenblas
# Target and source
TARGET = tsqr
SRC = tsqr.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SRC) $(LIBDIRS) $(LIBS)

clean:
	rm -f $(TARGET)
