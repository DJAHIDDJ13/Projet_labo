#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define N 1000 
#define NO_TRIES 10 

typedef long long int64;
typedef unsigned long long uint64;


void matmult_naive_no_opt(double **a, double **b, double **res) {
    int i, j, x;
    double acc;
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) { 
            acc = 0.0;
            for(x=0; x<N; x++) {
                acc += a[i][x] * b[x][j];
            }
            res[i][j] = acc;
        }
    }
}

void matmult_naive_tiling(double **a, double **b, double **res) {
    int i, j, x;
    double acc00, acc01, acc10, acc11;
    for(i=0; i<N-1; i+=2) {
        for(j=0; j<N-1; j+=2) { 
            acc00 = acc01 = acc10 = acc11 = 0.0;
            for(x=0; x<N; x++) {
                acc00 += a[i + 0][x] * b[x][j + 0];
                acc01 += a[i + 0][x] * b[x][j + 1];
                acc10 += a[i + 1][x] * b[x][j + 0];
                acc11 += a[i + 1][x] * b[x][j + 1];
            }
            res[i + 0][j + 0] = acc00;
            res[i + 0][j + 1] = acc01;
            res[i + 1][j + 0] = acc10;
            res[i + 1][j + 1] = acc11;
        }
    }
}

void matmult_naive_parallel(double **a, double **b, double **res) {
    int i, j, x;
    double acc;
#pragma omp parallel for shared(a, b, res) private(i, j, x, acc) schedule(static)
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) { 
            acc = 0.0;
            for(x=0; x<N; x++) {
                acc += a[i][x] * b[x][j];
            }
            res[i][j] = acc;
        }
    }
}

void matmult_naive_parallel_tiling(double **a, double **b, double **res) {
    int i, j, x;
    double acc00, acc01, acc10, acc11;
#pragma omp parallel for shared(a, b, res) private(i, j, x, acc00, acc01, acc10, acc11) schedule(static)
    for(i=0; i<N-1; i+=2) {
        for(j=0; j<N-1; j+=2) { 
            acc00 = acc01 = acc10 = acc11 = 0.0;
            for(x=0; x<N; x++) {
                acc00 += a[i + 0][x] * b[x][j + 0];
                acc01 += a[i + 0][x] * b[x][j + 1];
                acc10 += a[i + 1][x] * b[x][j + 0];
                acc11 += a[i + 1][x] * b[x][j + 1];
            }
            res[i + 0][j + 0] = acc00;
            res[i + 0][j + 1] = acc01;
            res[i + 1][j + 0] = acc10;
            res[i + 1][j + 1] = acc11;
        }
    }
}



/* int matmult_naive_no_opt(double **a, double **b, double **res) {
    int i, j, x;
    double acc00, acc01, acc10, acc11;
#pragma omp parallel shared(a, b, res) private(i, j, x, acc00)//, acc01, acc10, acc11)
    {
#pragma omp for schedule(static)
        for(i=0; i<N-1; i+=2) {
            for(j=0; j<N-1; j+=2) { 
                acc00 = 0.0;//acc01 = acc10 = acc11 = 0.0;
                for(x=0; x<N; x++) {
                    acc00 += a[i + 0][x] * b[x][j + 0];
                    acc01 += a[i + 0][x] * b[x][j + 1];
                    acc10 += a[i + 1][x] * b[x][j + 0];
                    acc11 += a[i + 1][x] * b[x][j + 1];
                }
                res[i + 0][j + 0] += acc00;
                res[i + 0][j + 1] += acc10;
                res[i + 1][j + 0] += acc01;
                res[i + 1][j + 1] += acc11;
            }
        }
    }
}
*/
uint64 get_time() {
    /* Linux */
    struct timeval tv;

    gettimeofday(&tv, NULL);

    uint64 ret = tv.tv_usec;
    /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
    ret /= 1000;

    /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
    ret += (tv.tv_sec * 1000);

    return ret;
}

void print_mat(double** res) {
    // written to stderr to ignore at execution
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            fprintf(stderr, "%g ", res[i][j]);
        }
        fprintf(stderr, "\n");
    }
}

void init_rand(double** a, double** b) {
    // init random a and b
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            a[i][j] = ((double) (rand()%1000000)) / 100000;
            b[i][j] = ((double) (rand()%1000000)) / 100000;
        }
    }
}

uint64 benchmark_matmult_func(double** a, double** b, double** res, void (*f)(double**,double**,double**)) {
    uint64 strt, end, elapsed, total_elapsed = 0;

    for (int i=0; i<NO_TRIES; i++) {
        // for benchmarking 
        strt = get_time();
        f(a, b, res);
        end = get_time();

        elapsed = end - strt;
        // printf("Elapsed time %llums\n", elapsed);
        
        total_elapsed += elapsed;
    }
    return ((double)total_elapsed)/NO_TRIES;
}

int main() {
    srand(time(NULL));

    printf ( "The number of processors available = %d\n", omp_get_num_procs());
    printf ( "The number of threads available    = %d\n\n", omp_get_max_threads());

    // init matrices
    double **a   = malloc(sizeof(double*) * N),
           **b   = malloc(sizeof(double*) * N),
           **res = malloc(sizeof(double*) * N);
    for(int i=0; i<N; i++) {
        a[i] = malloc(sizeof(double) * N);
        b[i] = malloc(sizeof(double) * N);
        res[i] = malloc(sizeof(double) * N);
    }
    
    // int a[N][N], b[N][N], res[N][N];
    
    init_rand(a, b);

    print_mat(a);
    print_mat(b); 
    
    uint64 average_elapsed;
    printf("Testing matrix multiplication for two %dx%d matrices, taking the average of %d tries\n\n", N, N, NO_TRIES);


    printf("Testing naive algorithm with no optimisations.. ");
    average_elapsed = benchmark_matmult_func(a, b, res, matmult_naive_no_opt);
    printf("Average time %llums\n", average_elapsed);
    print_mat(res);

    printf("Testing naive algorithm with nested loop tiling.. ");
    average_elapsed = benchmark_matmult_func(a, b, res, matmult_naive_tiling);
    printf("Average time %llums\n", average_elapsed);
    print_mat(res);

    printf("Testing naive algorithm with openmp parallelization.. ");
    average_elapsed = benchmark_matmult_func(a, b, res, matmult_naive_parallel);
    printf("Average time %llums\n", average_elapsed);
    print_mat(res);
    
    printf("Testing naive algorithm with openmp parallelization and tiling.. ");
    average_elapsed = benchmark_matmult_func(a, b, res, matmult_naive_parallel_tiling);
    printf("Average time %llums\n", average_elapsed);
    print_mat(res);
    return 0;
}
