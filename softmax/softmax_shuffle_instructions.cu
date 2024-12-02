#include <iostream>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#define WARP_SIZE 32

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




__inline__ __device__ void warp_reduce(float &max_num, float &sum) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float max_num_other = __shfl_down_sync(0xffffffff, max_num, offset);
        float sum_other = __shfl_down_sync(0xffffffff, sum, offset);

        // Combine
        float max_num_new = fmaxf(max_num_other, max_num);
        float sum_new = sum * expf(max_num - max_num_new)
                     + sum_other * expf(max_num_other - max_num_new);

        max_num = max_num_new;
        sum = sum_new;
    }
}



// online version of softmax, which uses parallleism across rows to compute softmax
__global__ void soft_max_kernel_shuffle(const float* matrix, float* matrix_out, size_t columns, size_t N) {
    
    extern __shared__ float sdata[]; // this will store the warp level data
    size_t row_index = blockIdx.x;
    size_t id = threadIdx.x;
    size_t range = (columns + blockDim.x-1)/blockDim.x;
    size_t block_dim = blockDim.x;
    float max_num = std::numeric_limits<float>::lowest();
    float sum = 0;
    if(id < block_dim){ 
        max_num = matrix[row_index * columns + id];
        sum = 1.0f; 
        for(int i = 1; i < range; i++){
            int index = row_index * columns  + i*block_dim + id;
            if(index < (row_index + 1) * columns){
                float new_max_num = fmaxf(max_num, matrix[index]);
                sum = sum * expf(max_num - new_max_num) + expf(matrix[index] - new_max_num);
                max_num = new_max_num;
            }
        }
    }

    warp_reduce(max_num, sum);

    int warp_id = id/32; // this symbolises the warp id of the thread
    int warp_id_rem = id % 32;

    if(warp_id_rem == 0){
        sdata[2*warp_id] = max_num;
        sdata[2*warp_id+1] = sum;
    }
    __syncthreads();

    max_num = std::numeric_limits<float>::lowest();
    sum = 0;
    if(warp_id_rem < block_dim/WARP_SIZE){
        max_num = sdata[2*warp_id_rem];
        sum = sdata[2*warp_id_rem+1];
    }

    warp_reduce(max_num, sum);

    max_num =  __shfl_sync(0xFFFFFFFF, max_num, 0);
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);

    if(id < block_dim){ 
        for(int i = 0; i < range; i++){
            int index = row_index * columns  + i*block_dim + id;
            if(index < (row_index + 1) * columns){
                matrix_out[index] = expf(matrix[index] - max_num)/sum;
            }
        }
    }
}

int main() {
    printCurrentDeviceProperties();
    size_t rows = 1024;
    size_t columns = 1024 * 64;
    size_t size = rows * columns * sizeof(float);
    // ideally this should be configured on the basis of the column size of the matrix, but the code is written agnostic to block_size. This should be a multiple of 32 ans also block_size/32 should be less than the warp size of 32
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
    soft_max_kernel_shuffle<<<rows, block_size, (block_size/WARP_SIZE) * 2 * sizeof(float)>>>(matrix, matrix_out, columns, count);
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
