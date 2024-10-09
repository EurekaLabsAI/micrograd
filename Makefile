CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lm

all: micrograd #plot

micrograd: micrograd.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	./$@

# plot: micrograd
# 	./micrograd
# 	gnuplot plot_script.gp

# clean:
# 	rm -f micrograd plot_data.txt plot.png
clean:
	rm -f micrograd

# .PHONY: all clean plot