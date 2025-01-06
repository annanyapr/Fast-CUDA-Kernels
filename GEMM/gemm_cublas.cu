#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// Matrix dimensions
static const int SIZE = 4096; // Square matrices: SIZE x SIZE

// Function to randomize a matrix
void randomize_matrix(float *mat, int N) {
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01f * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.f);
        mat[i] = tmp;
    }
}

// Function to print current CUDA device properties
void printCurrentDeviceProperties() {
    int currentDevice;
    cudaGetDevice(&currentDevice);  // Get current device ID

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, currentDevice);  // Get device properties

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

    // Matrix dimensions: square matrices of size SIZE x SIZE.
    int m = SIZE, n = SIZE, k = SIZE;
    const float alpha = 0.5f, beta = 3.0f;

    // Allocate host memory for matrices A, B, C
    float *A = (float *)malloc(sizeof(float) * m * k);
    float *B = (float *)malloc(sizeof(float) * k * n);
    float *C = (float *)malloc(sizeof(float) * m * n);

    if (!A || !B || !C) {
        std::cerr << "Host memory allocation error.\n";
        exit(EXIT_FAILURE);
    }

    // Initialize matrices with random values
    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);
    randomize_matrix(C, m * n);

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void **)&d_A, sizeof(float) * m * k);
    cudaMalloc((void **)&d_B, sizeof(float) * k * n);
    cudaMalloc((void **)&d_C, sizeof(float) * m * n);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    // Create a cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // --- Timing using CUDA events ---
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    // IMPORTANT: cuBLAS assumes column-major storage.
    // To compute C = alpha*A*B + beta*C using row-major matrices,
    // one way is to swap the order and parameters:
    //   C^T = alpha * B^T * A^T + beta * C^T
    // Therefore, call cublasSgemm with transposed operands.
    cublasStatus_t stat = cublasSgemm(handle,
                                      CUBLAS_OP_N,  // op(B) not transposed
                                      CUBLAS_OP_N,  // op(A) not transposed
                                      n,  // number of rows of matrix op(B) and C^T
                                      m,  // number of columns of matrix op(A) and C^T
                                      k,  // shared dimension
                                      &alpha,
                                      d_B, n,  // B: dimensions (k x n) in row-major becomes (n x k) in col-major
                                      d_A, k,  // A: dimensions (m x k) in row-major becomes (k x m) in col-major
                                      &beta,
                                      d_C, n); // C: dimensions (m x n) in row-major becomes (n x m) in col-major

    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS SGEMM failed\n";
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    elapsed_time /= 1000.0f;  // convert to seconds

    long flops = 2L * m * n * k;
    printf("cuBLAS SGEMM: elapsed time = %7.6f s, performance = %7.1f GFLOPS.\n",
           elapsed_time, (flops * 1e-9f) / elapsed_time);

    // Copy result matrix C from device to host
    cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    // Optionally, you can print out a few elements to check results
    // printf("First element of C = %f\n", C[0]);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}
