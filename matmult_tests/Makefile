CC=gcc
PARAMS=-g -Wall -fopenmp
exec: matmult_simple
	./matmult_simple_O0 2> /dev/null
	./matmult_simple_O1 2> /dev/null
	./matmult_simple_O2 2> /dev/null
	./matmult_simple_O3 2> /dev/null

matmult_simple: matmult_simple.c
	$(CC) $(PARAMS) -O0 -o matmult_simple_O0 $<
	$(CC) $(PARAMS) -O1 -o matmult_simple_O1 $<
	$(CC) $(PARAMS) -O2 -o matmult_simple_O2 $< # default level
	$(CC) $(PARAMS) -O3 -o matmult_simple_O3 $<

clean:
	rm -f matmult_simple_O*
