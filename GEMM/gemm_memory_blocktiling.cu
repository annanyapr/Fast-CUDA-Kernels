#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define VECTOR_INDEX(i, j, ld) ((i) * (ld) + (j))

static const int BM = 64; 
static const int BN = 64;  
static const int BK = 8;   
static const int TM = 8;
static const int TN = 8;


void randomize_matrix(float *mat, int N);


__global__ void gemm_2D_blocktiling(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C)
{
    int block_row = blockIdx.x;  
    int block_col = blockIdx.y; 

    int tid  = threadIdx.x;
    int trow_block = tid / (BN / TN);
    int tcol_block = tid % (BN / TN);

    __shared__ float mem_A[BM * BK];  // tile of A (size BM×BK)
    __shared__ float mem_B[BK * BN];  // tile of B (size BK×BN)

    int number_threads_blocktile = (BM*BN)/(TN*TM);

    // starting index which a given thread id will load
    int inner_col_a = tid % BK;  
    int inner_row_a = tid / BK;  
    int inner_col_b = tid % BN;   
    int inner_row_b = tid / BN;   

    int stride_a = number_threads_blocktile/BK;
    int stride_b = number_threads_blocktile/BN;
    
    float accum[TM * TN] = {0.0f}; // accumulates results here
    float a_reg[TM] = {0.0f}; 
    float b_reg[TN] = {0.0f};

    // for easier indexing move the blocks
    A += block_row * BM * k;
    B += block_col * BN;
    C += block_row * BM * n + block_col * BN;

    // TM * TN is a 8 x 8 block which will be calculate by each thread
    for (int kb = 0; kb < k; kb += BK) {

        for(int row_offset = 0; row_offset < BM; row_offset+=stride_a){
            mem_A[VECTOR_INDEX(inner_row_a + row_offset, inner_col_a, BK)] = A[VECTOR_INDEX(inner_row_a + row_offset, inner_col_a, k)];
        }

        for(int row_offset = 0; row_offset < BK; row_offset+=stride_b){
            mem_B[VECTOR_INDEX(inner_row_b + row_offset, inner_col_b, BN)] = B[VECTOR_INDEX(inner_row_b + row_offset, inner_col_b, n)];
        }

        __syncthreads();

        // now we have loaded all the data, now just load in register and calculate 
        
        for(int idx = 0; idx < BK; idx++){

            // load into register A 
            for(int i = 0; i < TM; i++){
                a_reg[i] = mem_A[VECTOR_INDEX(trow_block * TM + i, idx, BK)];
            }

            // load into register B
            for(int i = 0; i < TN; i++){
                b_reg[i] = mem_B[VECTOR_INDEX(idx, tcol_block * TN + i, BN)];
            }


            for(int idxx = 0; idxx < TM; idxx++){
                for(int idxy = 0; idxy < TN; idxy++){
                    accum[VECTOR_INDEX(idxx, idxy, TN)] += a_reg[idxx] * b_reg[idxy];
                }
            }

        }
        
        __syncthreads();
        A += BK; 
        B += BK*n;

    }


    for(int idxx = 0; idxx < TM; idxx++){
        for(int idxy = 0; idxy < TN; idxy++){
            C[VECTOR_INDEX(trow_block* TM+ idxx, tcol_block*TN+ idxy, n)] = alpha * accum[VECTOR_INDEX(idxx, idxy, TN)] + beta * C[VECTOR_INDEX(trow_block* TM+ idxx, tcol_block*TN+ idxy, n)];
        }
    }

}


void run_kernel(int M, int N, int K,
                   float alpha,
                   float *dA,
                   float *dB,
                   float beta,
                   float *dC)
{
    dim3 gridDim( CEIL_DIV(M, BM), CEIL_DIV(N, BN) );
    int blockSize = (BM*BN)/(TM*TN);
    dim3 blockDim(blockSize);

    gemm_2D_blocktiling<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
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