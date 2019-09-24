#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define mat_t int

typedef struct {
	int n, m;
	mat_t* arr;
} Mat;


mat_t get_elem(Mat m, int x, int y) {
	return m.arr[y * m.n + x];
}

void set_elem(Mat m, int x, int y, mat_t val) {
	m.arr[y * m.n + x] = val;
}

Mat mult_mat(Mat a, Mat b) {
	if(a.m != b.n) {
		fprintf(stderr, "Mismatched matrices: %dx%d and %dx%d\n", a.n, a.m, b.n, b.m);
		exit(1);
	}
	int N = a.n, M = b.m; // dimensions of resulting matrix
	Mat res = {
		.n = N,
	       	.m = M,
	       	.arr = malloc(sizeof(mat_t) * N * M)
	};
	
	for(int i=0; i<N; i++) {
		for(int j=0; j<M; j++) { 
			mat_t cumul = 0;
			for(int x=0; x<a.m; x++) {
				cumul += get_elem(a, i, x) * get_elem(b, x, j);
			}
			set_elem(res, i, j, cumul);
		}
	}

	return res;
}

void print_mat(Mat m) {
	fprintf(stderr, "Matrix [%dx%d]\n", m.n, m.m);
	for(int i=0; i<m.n; i++) {
		for(int j=0; j<m.m; j++) {
			fprintf(stderr, "%d ", get_elem(m, i, j));
		}
		fprintf(stderr, "\n");
	}
}

// Generates the unit matrix 
Mat gen_unit_mat(int N, int M) {
	Mat res = {
		.n = N,
	       	.m = M,
	       	.arr = malloc(sizeof(mat_t) * N * M)
	};
	memset(res.arr, 0, sizeof(mat_t) * N * M);
	
	int min_dim = (N < M)? N:M;
	for(int i=0; i<min_dim; i++) {
		set_elem(res, i, i, 1);
	}
	return res;	
}

// matrix with random uniform values between min_val and max_val (won't work for mat_t different than int)
Mat gen_runif_mat(int N, int M, int min_val, int max_val) {
	Mat res = gen_unit_mat(N, M);
	for(int i=0; i<N; i++) {
		for(int j=0; j<M; j++) {
			set_elem(res, i, j, min_val + rand()%max_val);
		}
	}
	return res;
}

int main() {
	srand(time(NULL)); // use current time as random seed
	int REPEATS = 100;
	double elapsed = 0.0;

	for(int i=0; i<REPEATS; i++) {
		int N = 100;
		// generate 2 random NxN matrices
		int min_val = 0, max_val = 20;
		Mat a = gen_runif_mat(N, N, min_val, max_val),
		    b = gen_runif_mat(N, N, min_val, max_val);
		print_mat(a);
		print_mat(b);

		// used to benchmark the mult_mat function
		clock_t strt, end;

		strt = clock(); // start time
		Mat c = mult_mat(a, b);
		end = clock(); // end time

		elapsed += ((double) (end - strt)) / CLOCKS_PER_SEC; // calculate elapsed time in seconds
	       
		print_mat(c);
	
		// cleanup
		free(c.arr);
		free(a.arr);
		free(b.arr);
	}
	printf("Average time elapsed %gs\n", elapsed / REPEATS);

	return 0;
}
