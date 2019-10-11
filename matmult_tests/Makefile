CC=gcc
PARAMS=-pg -Wall -fopenmp

matmult_simple: matmult_simple.c
	$(CC) $(PARAMS) -O0 -o matmult_simple_O0 $<
	$(CC) $(PARAMS) -O1 -o matmult_simple_O1 $<
	$(CC) $(PARAMS) -O2 -o matmult_simple_O2 $<
	$(CC) $(PARAMS) -O3 -o matmult_simple_O3 $<

exec: matmult_simple
	./matmult_simple_O0 -O0 2> /dev/null 
	./matmult_simple_O1 -O1 2> /dev/null
	./matmult_simple_O2 -O2 2> /dev/null
	./matmult_simple_O3 -O3 2> /dev/null

clean:
	rm -f matmult_simple_O*
