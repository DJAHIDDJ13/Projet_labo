#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define N 100
#define NO_TRIES 1

typedef long long int64;
typedef unsigned long long uint64;


void matmult_naive_no_opt(float **a, float **b, float **res) {
    int i, j, x;
    float acc;
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

// DOES NOT WORK FOR ODD Ns TODO
void matmult_naive_tiling(float **a, float **b, float **res) {
    int ii, i, j, kk, k;
    float acc00, acc01, acc10, acc11;

    int ib = 30, kb = 30, ilim, klim, lim;
    for (ii = 0; ii < N; ii += ib) {
        for (kk = 0; kk < N; kk += kb) {
            for (j=0; j < N; j += 2) {
                lim = ii+ib;
                ilim = (lim > N)? N: lim;
                for(i = ii; i < ilim; i += 2 ) {
                    if (kk == 0)
                        acc00 = acc01 = acc10 = acc11 = 0;
                    else {
                        acc00 = res[i + 0][j + 0];
                        acc01 = res[i + 0][j + 1];
                        acc10 = res[i + 1][j + 0];
                        acc11 = res[i + 1][j + 1];
                    }
                    lim = kk+kb;
                    klim = (lim > N)? N: lim;
                    for (k = kk; k < klim; k++) {
                        acc00 += b[k][j + 0] * a[i + 0][k];
                        acc01 += b[k][j + 1] * a[i + 0][k];
                        acc10 += b[k][j + 0] * a[i + 1][k];
                        acc11 += b[k][j + 1] * a[i + 1][k];
                    }
                    res[i + 0][j + 0] = acc00;
                    res[i + 0][j + 1] = acc01;
                    res[i + 1][j + 0] = acc10;
                    res[i + 1][j + 1] = acc11;
                }
            }
        }
    }
}

void matmult_naive_parallel(float **a, float **b, float **res) {
    int i, j, x;
    float acc;
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

// DOES NOT WORK FOR ODD Ns TODO
void matmult_naive_parallel_tiling(float **a, float **b, float **res) {
    int ii, i, j, kk, k;
    float acc00, acc01, acc10, acc11;   

    int ib = 30, kb = 30;
#pragma omp parallel for shared(a, b, res, ib, kb) private(i, ii, j, k, kk, acc00, acc01, acc10, acc11) schedule(static)
    for (ii = 0; ii < N; ii += ib) {
        for (kk = 0; kk < N; kk += kb) {
            for (j=0; j < N; j += 2) {
                int ilim = (ii+ib > N)? N: ii+ib;
                for(i = ii; i < ilim; i += 2 ) {
                    if (kk == 0)
                        acc00 = acc01 = acc10 = acc11 = 0;
                    else {
                        acc00 = res[i + 0][j + 0];
                        acc01 = res[i + 0][j + 1];
                        acc10 = res[i + 1][j + 0];
                        acc11 = res[i + 1][j + 1];
                    }
                    int klim = (kk+kb > N)? N: kk+kb;
                    for (k = kk; k < klim; k++) {
                        acc00 += b[k][j + 0] * a[i + 0][k];
                        acc01 += b[k][j + 1] * a[i + 0][k];
                        acc10 += b[k][j + 0] * a[i + 1][k];
                        acc11 += b[k][j + 1] * a[i + 1][k];
                    }
                    res[i + 0][j + 0] = acc00;
                    res[i + 0][j + 1] = acc01;
                    res[i + 1][j + 0] = acc10;
                    res[i + 1][j + 1] = acc11;
                }
            }
        }
    }
}

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

// to verify the results in matlab/octave
void print_mat(float** res, const char* var_name) {
    // written to stderr to ignore at execution
    fprintf(stderr, "%s = [", var_name);
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            fprintf(stderr, "%g ", res[i][j]);
        }
        fprintf(stderr, "%s\n",(i==N-1)?"":";");
    }
    fprintf(stderr, "];\n");
}

void init_rand(float** a, float** b) {
    // init random a and b
    int max_val = 10;
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            a[i][j] = ((float) rand())/((float) RAND_MAX) * max_val;
            b[i][j] = ((float) rand())/((float) RAND_MAX) * max_val;
        }
    }
}

uint64 benchmark_matmult_func(float** a, float** b, float** res, void (*f)(float**,float**,float**)) {
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
    return ((float)total_elapsed)/NO_TRIES;
}

int main(int argc, char** argv) {
    srand(time(NULL));
    if(argc != 2) {
        printf("Write the optimisation level as first argument\n");
        exit(1);
    }
    // going to assume the first argument is the optimisation level
    const char* opt_level = argv[1];

    FILE* csv_out = fopen("output.csv", "a");

    omp_set_num_threads(4);

    printf ( "The number of processors available = %d\n", omp_get_num_procs());
    printf ( "The number of threads available    = %d\n\n", omp_get_max_threads());

    // init matrices
    float **a   = malloc(sizeof(float*) * N),
           **b   = malloc(sizeof(float*) * N),
           **res = malloc(sizeof(float*) * N);
    for(int i=0; i<N; i++) {
        a[i] = malloc(sizeof(float) * N);
        b[i] = malloc(sizeof(float) * N);
        res[i] = malloc(sizeof(float) * N);
    }

    // int a[N][N], b[N][N], res[N][N];

    init_rand(a, b);

    print_mat(a, "A");
    print_mat(b, "B"); 

    uint64 average_elapsed;
    printf("Testing matrix multiplication for two %dx%d matrices, taking the average of %d tries\n\n", N, N, NO_TRIES);

    printf("Testing naive algorithm with no optimisations.. ");
    average_elapsed = benchmark_matmult_func(a, b, res, matmult_naive_no_opt);
    printf("Average time %llums\n", average_elapsed);
    print_mat(res, "C1");
    fprintf(csv_out, "%s:%d:no opt:%lld\n", opt_level, N, average_elapsed);

    printf("Testing naive algorithm with nested loop tiling.. ");
    average_elapsed = benchmark_matmult_func(a, b, res, matmult_naive_tiling);
    printf("Average time %llums\n", average_elapsed);
    print_mat(res, "C2");
    fprintf(csv_out, "%s:%d:with opt:%lld\n", opt_level, N, average_elapsed);

    printf("Testing naive algorithm with openmp parallelization.. ");
    average_elapsed = benchmark_matmult_func(a, b, res, matmult_naive_parallel);
    printf("Average time %llums\n", average_elapsed);
    print_mat(res, "C3");
    fprintf(csv_out, "%s:%d:parallel:%lld\n", opt_level, N, average_elapsed);
    
    printf("Testing naive algorithm with openmp parallelization and tiling.. ");
    average_elapsed = benchmark_matmult_func(a, b, res, matmult_naive_parallel_tiling);
    printf("Average time %llums\n", average_elapsed);
    print_mat(res, "C4");
    fprintf(csv_out, "%s:%d:with opt+parallel:%lld\n", opt_level, N, average_elapsed);

    fclose(csv_out);
    return 0;
}
