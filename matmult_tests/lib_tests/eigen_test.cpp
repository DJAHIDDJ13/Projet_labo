#include <iostream>
#include <Eigen/Dense>
#include <chrono>

#define N 2000 

using namespace Eigen;
using namespace std::chrono;

int main() {
    MatrixXcf a = MatrixXcf::Random(N, N), b = MatrixXcf::Random(N, N);
    high_resolution_clock::time_point strt, end;
    
    strt = high_resolution_clock::now();
    MatrixXcf res = a * b;
    end = high_resolution_clock::now();

//    std::cout << "A:\n" << a << std::endl;
//    std::cout << "\nB:\n" << b << std::endl;
//    std::cout << "\nres\n" << res << std::endl;
    
    duration<double, std::milli> time_span = end - strt;
    
    std::cout << "Time taken: " << time_span.count() << "ms" << std::endl; 
}
