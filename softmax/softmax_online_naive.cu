#include <iostream>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>

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


// this is the online version of softmax
__global__ void soft_max_kernel_naive(const float* matrix, float* matrix_out, size_t columns, size_t N) {
    
    size_t row_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t starting_location = row_index * columns;

    if(starting_location < N){
        float max_num = matrix[starting_location];
        float sum  = 1.0f; 
        for(int i = 1; i < columns; i++){
            float new_max_num = fmaxf(max_num, matrix[starting_location+i]);
            // we will compute the max and sum in one pass
            sum = sum * expf(max_num - new_max_num) + expf(matrix[starting_location+i] - new_max_num);
            max_num = new_max_num;
        }
        for(int i = 0; i < columns; i++){
            matrix_out[starting_location+i] = expf(matrix[starting_location+i] - max_num)/sum;
        }
    }
}


// this is the online version of softmax which uses shared memory
__global__ void soft_max_kernel_parallel(const float* matrix, float* matrix_out, size_t columns, size_t N) {
    
    size_t row_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t starting_location = row_index * columns;

    if(starting_location < N){
        float max_num = matrix[starting_location];
        float sum  = 1.0f; 
        for(int i = 1; i < columns; i++){
            float new_max_num = fmaxf(max_num, matrix[starting_location+i]);
            // we will compute the max and sum in one pass
            sum = sum * expf(max_num - new_max_num) + expf(matrix[starting_location+i] - new_max_num);
            max_num = new_max_num;
        }
        for(int i = 0; i < columns; i++){
            matrix_out[starting_location+i] = expf(matrix[starting_location+i] - max_num)/sum;
        }
    }
}

int main() {
    printCurrentDeviceProperties();
    size_t rows = 1024;
    size_t columns = 1024 * 64;
    size_t size = rows * columns * sizeof(float);
    int block_size = 256;
    int count = rows * columns;
    float* matrix_cpu = (float*) malloc(size) ;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Seed the random number generator
    srand(time(NULL));

    // Fill the array with random numbers from -5 to 5
    for (size_t i = 0; i < rows * columns; ++i) {
        matrix_cpu[i] = -5 + ((float)rand() / RAND_MAX) * 10;
    }

    float* matrix;
    cudaMalloc(&matrix, rows * columns * sizeof(float));
    cudaMemcpy(matrix, matrix_cpu, size, cudaMemcpyHostToDevice);
    
    // the output matrix
    float* matrix_out;
    cudaMalloc(&matrix_out, rows * columns * sizeof(float));
    
    cudaEventRecord(start);
    soft_max_kernel_naive<<<(rows+block_size-1)/block_size, block_size>>>(matrix, matrix_out, columns, count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "Softmax kernel took " << elapsed << " ms\n";

    float* matrix_cpu_out = (float*) malloc(1024*64*4) ;
    cudaMemcpy(matrix_cpu_out, matrix_out, 1024*64*4, cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for (int i = 0; i < 1024*64; ++i)
        sum += matrix_cpu_out[i];
    std::cout << "The sum of the first column is: '" << sum << std::endl;
    cudaFree(matrix);
    cudaFree(matrix_out);
    free(matrix_cpu_out);
    free(matrix_cpu);
    return 0;
}
