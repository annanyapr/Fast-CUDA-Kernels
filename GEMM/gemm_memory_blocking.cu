#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define BLOCK_SIZE 32
#define BLOCKSIZE 32
#define VECTOR_INDEX(i, j, n) (i * n + j) 


void randomize_matrix(float *mat, int N);

// online version of softmax, which uses parallleism across rows to compute softmax
// C = alpha * A * B + beta * C; A is of dimension (m, k), B is of dimension (k, n)
__global__ void gemm_memory_blocking(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    uint x = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    uint y = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;
    // shared memory
    __shared__ float mem_A[BLOCK_SIZE * BLOCK_SIZE]; 
    __shared__ float mem_B[BLOCK_SIZE * BLOCK_SIZE]; 

    uint start_x = blockIdx.x * BLOCK_SIZE;
    uint start_y = blockIdx.y * BLOCK_SIZE;
    uint block_x = threadIdx.x / BLOCK_SIZE;
    uint block_y = threadIdx.x % BLOCK_SIZE;

    float tmp = 0.0f;
    for(int block_index = 0; block_index < k; block_index += BLOCK_SIZE){
        mem_A[VECTOR_INDEX(block_x, block_y, BLOCK_SIZE)] = A[VECTOR_INDEX(start_x + block_x, block_index + block_y, k)]; 
        mem_B[VECTOR_INDEX(block_x, block_y, BLOCK_SIZE)] = B[VECTOR_INDEX(block_index + block_x, start_y + block_y, n)];

        __syncthreads();

        // now lets execute the dot product
        for(int i = 0; i < BLOCK_SIZE; i++){
            tmp += mem_A[VECTOR_INDEX(block_x, i, BLOCK_SIZE)] * mem_B[VECTOR_INDEX(i, block_y, BLOCK_SIZE)];
        }
        __syncthreads();
    }
    C[VECTOR_INDEX(x, y, n)] = alpha * tmp + beta * C[VECTOR_INDEX(x, y, n)];
}


// Kernel launcher function 
void run_kernel(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    dim3 grid_dim = dim3(m/BLOCK_SIZE, n/BLOCK_SIZE);
    gemm_memory_blocking<<<grid_dim, BLOCK_SIZE * BLOCK_SIZE>>>(m, n, k, alpha, A, B, beta, C);
    cudaError_t err = cudaGetLastError();
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
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Kernel execution
    cudaEventRecord(start);
    for (int j = 0; j < repeat_times; j++) {
        run_kernel(size, size, size, alpha, dA, dB, beta, dC);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&elapsed_time, start, end);
    elapsed_time /= 1000.0; // Convert to seconds

    long flops = 2L * size * size * size;
    printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS.\n",
           elapsed_time / repeat_times,
           (repeat_times * flops * 1e-9) / elapsed_time);

    cudaMemcpy(C, dC, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

    // Print the first element computed by GPU and CPU
    // printf("CPU result for the first element: %f\n", cpu_result);
    // printf("GPU result for the first element: %f\n", C[0]);


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

// void randomize_matrix(float *mat, int N) {
//     for (int i = 0; i < N; i++) {
//         // Fill entries with 1..8 in a repeating pattern
//         mat[i] = 1 + (i*i % 7);
//     }
// }