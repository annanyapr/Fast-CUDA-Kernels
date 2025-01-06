#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

void randomize_matrix(float *mat, int N) ;



// online version of softmax, which uses parallleism across rows to compute softmax
// C = alpha * A * B + beta * C; A is of dimension (m, k), B is of dimension (k, n)
__global__ void gemm_naive(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < m && y < n){
        float tmp = 0;
        for(int dim = 0; dim < k; dim++){
            tmp += A[x*k + dim] * B[dim*n + y];
        }
        C[x*n+y] = alpha * tmp + beta * C[x*n+y];
    }
}




// Kernel launcher function 
void run_kernel(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    dim3 grid_dim = dim3(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    gemm_naive<<<grid_dim, dim3(32, 32)>>>(m, n, k, alpha, A, B, beta, C);
}


float cpu_compute_first_element(float *A, float *B, float alpha, float beta, float *C, int size) {
    float result = 0.0;
    for (int i = 0; i < size; i++) {
        result += A[i] * B[i * size];
    }
    result = alpha * result + beta * C[0];
    return result;
}




void printCurrentDeviceProperties() {
    int currentDevice;
    cudaGetDevice(&currentDevice);  // Get the current device ID

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, currentDevice);  // Get properties of the current device

    std::cout << "Currently Active CUDA Device: " << currentDevice << "\n";
    std::cout << "Device Name: " << prop.name << "\n";
    std::cout << "CUDA Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
    std::cout << "Warp Size: " << prop.warpSize << "\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
}

int main() {
    printCurrentDeviceProperties();
    // Matrix size
    const int size = 4096;
    const float alpha = 0.5, beta = 3.0; // GEMM parameters, C = α*AB + β*C

    const int repeat_times = 1;
    // Host and device matrices
    float *A = (float *)malloc(sizeof(float) * size * size);
    float *B = (float *)malloc(sizeof(float) * size * size);
    float *C = (float *)malloc(sizeof(float) * size * size);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc((void **)&dA, sizeof(float) * size * size);
    cudaMalloc((void **)&dB, sizeof(float) * size * size);
    cudaMalloc((void **)&dC, sizeof(float) * size * size);

    // Initialize matrices
    randomize_matrix(A, size * size);
    randomize_matrix(B, size * size);
    randomize_matrix(C, size * size);

        // Naive CPU computation for the first element
    float cpu_result = cpu_compute_first_element(A, B, alpha, beta, C, size);

    cudaMemcpy(dA, A, sizeof(float) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(float) * size * size, cudaMemcpyHostToDevice);

    // Timing events
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Kernel execution
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
        run_kernel(size, size, size, alpha, dA, dB, beta, dC);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.0; // Convert to seconds

    long flops = 2L * size * size * size;
    printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS.\n",
           elapsed_time / repeat_times,
           (repeat_times * flops * 1e-9) / elapsed_time);

    cudaMemcpy(C, dC, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

    // Print the first element computed by GPU and CPU
    printf("CPU result for the first element: %f\n", cpu_result);
    printf("GPU result for the first element: %f\n", C[0]);


    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}


// Randomize matrix function
void randomize_matrix(float *mat, int N) {
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}