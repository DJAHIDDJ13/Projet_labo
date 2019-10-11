#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <time.h>

int32_t SIMD_sum(const int32_t a[], const int n)
{
    __m128i vsum = _mm_set1_epi32(0);       // initialise vector of four partial 32 bit sums
    int32_t sum;
    int i;

    for (i = 0; i < n; i += 4)
    {
        __m128i v = _mm_load_si128((_m128i)(a + i));  // load vector of 4 x 32 bit values
        vsum = _mm_add_epi32(vsum, v);      // accumulate to 32 bit partial sum vector
    }
    // horizontal add of four 32 bit partial sums and return result
    vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 8));
    vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 4));
    sum = _mm_cvtsi128_si32(vsum);
    return sum;
}

int naive_sum(const int arr[], const int n) {
    int sum = 0;
    for(int i=0; i<n; i++) {
        sum += arr[i];
    }
    return sum;
}

int main(int argc, char *argv[]) {
    int n = 128;
    const int32_t arr[128];
    
    int sum = 0;
    
    clock_t strt, end;
    double elapsed;
    
    strt = clock();
    sum = naive_sum(arr, n);
    end = clock();
    
    elapsed = ((double) (end-strt)) / CLOCKS_PER_SEC;

    printf("[NAIVE]sum = %d; elapsed = %gs\n", sum, elapsed);

    strt = clock();
    sum = SIMD_sum(arr, n);
    end = clock();
    
    elapsed = ((double) (end-strt)) / CLOCKS_PER_SEC;

    printf("[SIMD]sum = %d; elapsed = %gs\n", sum, elapsed);
    
    return 0;
}
