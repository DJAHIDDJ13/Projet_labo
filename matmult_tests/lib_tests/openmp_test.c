#include <omp.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char *argv[]) {
    long long sum = 0;

    clock_t strt, end;
    double elapsed;
    strt = clock();
#pragma omp parallel for shared(sum)
    for(long i=0; i<1000000000; i++) {
        sum ++;
    }
    end = clock();
    elapsed = ((double) (end-strt)) / CLOCKS_PER_SEC;

    printf("[PARALLEL]sum = %d; elapsed = %gs\n", sum, elapsed);

    sum = 0;
    strt = clock();
    end = clock();
    for(long i=0; i<1000000000; i++) {
        sum ++;
    }
    elapsed = ((double) (end-strt)) / CLOCKS_PER_SEC;


    printf("[NOT_PARALLEL]sum = %d; elapsed = %gs\n", sum, elapsed);

    return 0;
}
