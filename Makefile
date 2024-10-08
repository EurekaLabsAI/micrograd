CC = gcc
CFLAGS = -Wall -Wextra

all: run

micrograd: micrograd.c
	$(CC) $(CFLAGS) $< -o $@

run: micrograd
	./$<

clean:
	rm -f micrograd