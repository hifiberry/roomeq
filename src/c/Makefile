CC = gcc
# Generic build flags (remove ARM-specific flags for portability)
CFLAGS  = -O3 -Wall
LD = ld
LDFLAGS = -lm -lasound

all: bin

OBJ = analyze.o
BIN = fft-analyse

bin: $(OBJ)
	$(CC) $(CFLAGS) -o $(BIN) $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm -rf $(BIN) *.o
