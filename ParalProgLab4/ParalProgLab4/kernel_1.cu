#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups.h>
#include <cuda/atomic>

namespace cg = cooperative_groups;

constexpr int TILE_SIZE = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int BLOCK_COLS = 32;

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): "
            << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCudaDevices() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
    }
    else {
        std::cout << "Found " << deviceCount << " CUDA-capable device(s):" << std::endl;

        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            std::cout << "  Device " << i << ": " << props.name << std::endl;
            std::cout << "    Compute capability: " << props.major << "." << props.minor << std::endl;
        }
    }
}


void write_matrix(const std::vector<std::vector<uint32_t>>& matrix,
    const std::string& filename) {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);

    for (const auto& row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j] << (j + 1 < row.size() ? "," : "\n");
        }
    }
}

// Ядро CUDA для умножения матриц
__global__ void matrixMultiplySharedKernel(uint32_t* A, uint32_t* B, uint32_t* C,
    int rowsA, int colsA, int colsB) {
    __shared__ uint32_t sA[16][16];
    __shared__ uint32_t sB[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sum = 0;

    for (int tile = 0; tile < (colsA + blockDim.x - 1) / blockDim.x; ++tile) {
        // Загрузка тайлов в shared memory
        int tiledCol = tile * blockDim.x + threadIdx.x;
        int tiledRow = tile * blockDim.y + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < rowsA && tiledCol < colsA)
            ? A[row * colsA + tiledCol] : 0;
        sB[threadIdx.y][threadIdx.x] = (tiledRow < colsA && col < colsB)
            ? B[tiledRow * colsB + col] : 0;

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = sum;
    }
}

// Добавляем метод для транспонирования матрицы на устройстве
__global__ void transposeMatrixKernel(uint32_t* input, uint32_t* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

__global__ void matrixMultiplyTensorCore(const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    cg::thread_block b = cg::this_thread_block();
    __shared__ half sA[TILE_SIZE][TILE_SIZE + 1];  // +1 для выравнивания
    __shared__ half sB[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE_SIZE) {
        __half2 a_frag[4];
        __half2 b_frag[4];

        // Асинхронная загрузка тайлов
        if (row < M && t + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = __float2half(0.0f);

        if (col < N && t + threadIdx.y < K)
            sB[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = __float2half(0.0f);

        b.sync();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 8) {
            // Векторизованная загрузка через 128-битные инструкции
            *(uint4*)&a_frag[0] = *(uint4*)&sA[threadIdx.y][k];
            *(uint4*)&b_frag[0] = *(uint4*)&sB[k][threadIdx.x];

            // Tensor Core операции
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc += __half2float(a_frag[i].x) * __half2float(b_frag[i].x);
                acc += __half2float(a_frag[i].y) * __half2float(b_frag[i].y);
            }
        }
        b.sync();
    }

    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], acc);
    }
}

class GPUMatrixMultiplier {
    cudaStream_t stream_;
    half* d_A_, * d_B_;
    float* d_C_;
    size_t current_size_ = 0;

public:
    GPUMatrixMultiplier() {
        checkCudaError(cudaStreamCreate(&stream_), "Stream creation");
    }

    ~GPUMatrixMultiplier() {
        cudaFree(d_A_);
        cudaFree(d_B_);
        cudaFree(d_C_);
        cudaStreamDestroy(stream_);
    }

    void uploadMatrices(const std::vector<std::vector<uint32_t>>& A,
        const std::vector<std::vector<uint32_t>>& B) {
        const int M = A.size();
        const int K = A[0].size();
        const int N = B[0].size();

        // Конвертация во float16
        std::vector<half> h_A(M * K), h_B(K * N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < K; ++j)
                h_A[i * K + j] = __float2half(A[i][j]);

        for (int i = 0; i < K; ++i)
            for (int j = 0; j < N; ++j)
                h_B[i * N + j] = __float2half(B[i][j]);

        // Перевыделение памяти при необходимости
        if (M * K > current_size_) {
            cudaFree(d_A_);
            checkCudaError(cudaMalloc(&d_A_, M * K * sizeof(half)), "Alloc A");
        }
        if (K * N > current_size_) {
            cudaFree(d_B_);
            checkCudaError(cudaMalloc(&d_B_, K * N * sizeof(half)), "Alloc B");
        }
        if (M * N > current_size_) {
            cudaFree(d_C_);
            checkCudaError(cudaMalloc(&d_C_, M * N * sizeof(float)), "Alloc C");
        }

        // Асинхронная загрузка
        checkCudaError(cudaMemcpyAsync(d_A_, h_A.data(), M * K * sizeof(half),
            cudaMemcpyHostToDevice, stream_), "Copy A");
        checkCudaError(cudaMemcpyAsync(d_B_, h_B.data(), K * N * sizeof(half),
            cudaMemcpyHostToDevice, stream_), "Copy B");
        checkCudaError(cudaMemsetAsync(d_C_, 0, M * N * sizeof(float), stream_),
            "Memset C");
    }

    std::vector<std::vector<uint32_t>> multiply(int M, int N, int K) {
        dim3 block(TILE_SIZE, BLOCK_ROWS);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE);

        matrixMultiplyTensorCore << <grid, block, 0, stream_ >> > (d_A_, d_B_, d_C_, M, N, K);
        checkCudaError(cudaGetLastError(), "Kernel launch");

        // Асинхронная выгрузка
        std::vector<float> h_C(M * N);
        checkCudaError(cudaMemcpyAsync(h_C.data(), d_C_, M * N * sizeof(float),
            cudaMemcpyDeviceToHost, stream_), "Copy C");

        // Синхронизация
        checkCudaError(cudaStreamSynchronize(stream_), "Stream sync");

        // Конвертация в uint32_t
        std::vector<std::vector<uint32_t>> result(M, std::vector<uint32_t>(N));
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                result[i][j] = static_cast<uint32_t>(h_C[i * N + j]);

        return result;
    }
};


class Matrix {
    std::ifstream file;

public:
    Matrix(const std::string& filename) {
        file.open(filename);
        if (!file.is_open()) {
            throw std::runtime_error("File not open");
        }
    }

    ~Matrix() {
        if (file.is_open()) {
            file.close();
        }
    }

    std::vector<std::vector<uint32_t>> readFile() {
        std::vector<std::vector<uint32_t>> matrix;
        std::string line;

        while (std::getline(file, line)) {
            std::vector<uint32_t> row;
            std::stringstream ss(line);
            uint32_t value;
            while (ss >> value) {
                row.push_back(value);
                if (ss.peek() == ',' || ss.peek() == ' ') {
                    ss.ignore();
                }
            }
            matrix.push_back(row);
        }
        return matrix;
    }
    std::vector<std::vector<uint32_t>> transpose() {
        std::vector<std::vector<uint32_t>> matrix = readFile();

        // Если матрица пустая, возвращаем пустую матрицу
        if (matrix.empty() || matrix[0].empty()) {
            return {};
        }

        // Создаем новую матрицу с перевернутыми размерами
        std::vector<std::vector<uint32_t>> transposed(matrix[0].size(), std::vector<uint32_t>(matrix.size()));

        // Заполняем транспонированную матрицу
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }
    std::vector<std::vector<uint32_t>> mul_cuda_optimized(Matrix& other) {
        GPUMatrixMultiplier gpu_mult;
        auto matrix1 = readFile();
        auto matrix2 = other.readFile();

        if (matrix1[0].size() != matrix2.size())
            throw std::runtime_error("Incompatible dimensions");

        const int M = matrix1.size();
        const int K = matrix1[0].size();
        const int N = matrix2[0].size();

        gpu_mult.uploadMatrices(matrix1, matrix2);
        return gpu_mult.multiply(M, N, K);
    }

    // Умножение матриц с использованием CUDA
    std::vector<std::vector<uint32_t>> mul_cuda(Matrix& matrix) {
        std::vector<std::vector<uint32_t>> matrix1 = this->readFile();
        std::vector<std::vector<uint32_t>> matrix2 = matrix.readFile();

        if (matrix1.empty() || matrix2.empty() || matrix1[0].size() != matrix2.size()) {
            throw std::runtime_error("Matrix dimensions are not compatible for multiplication");
        }

        int rowsA = matrix1.size();
        int colsA = matrix1[0].size();
        int colsB = matrix2[0].size();

        // Подготовка данных для CUDA
        size_t sizeA = rowsA * colsA * sizeof(uint32_t);
        size_t sizeB = colsA * colsB * sizeof(uint32_t);
        size_t sizeC = rowsA * colsB * sizeof(uint32_t);

        uint32_t* h_A = new uint32_t[rowsA * colsA];
        uint32_t* h_B = new uint32_t[colsA * colsB];
        uint32_t* h_C = new uint32_t[rowsA * colsB];

        // Заполнение линейных массивов
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsA; ++j) {
                h_A[i * colsA + j] = matrix1[i][j];
            }
        }

        for (int i = 0; i < colsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                h_B[i * colsB + j] = matrix2[i][j];
            }
        }

        // Выделение памяти на устройстве
        uint32_t* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_B, sizeB);
        cudaMalloc(&d_C, sizeC);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Асинхронное копирование
        cudaMemcpyAsync(d_A, h_A, sizeA, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_B, h_B, sizeB, cudaMemcpyHostToDevice, stream);

        // Настройка параметров запуска ядра
        dim3 blockSize(8, 8);
        dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x,
            (rowsA + blockSize.y - 1) / blockSize.y);
        dim3 gridSizeTranspose((colsA + blockSize.x - 1) / blockSize.x,
            (colsB + blockSize.y - 1) / blockSize.y);
        // Транспонируем матрицу B на устройстве
        uint32_t* d_BT;
        cudaMalloc(&d_BT, sizeB);

        transposeMatrixKernel << <gridSizeTranspose, blockSize >> > (d_B, d_BT, colsA, colsB);

        // Используем оптимизированное ядро с shared memory
        matrixMultiplySharedKernel << <gridSize, blockSize >> > (d_A, d_BT, d_C, rowsA, colsA, colsB);

        // Асинхронное копирование обратно
        cudaMemcpyAsync(h_C, d_C, sizeC, cudaMemcpyDeviceToHost, stream);

        // Преобразование результата обратно в вектор векторов
        std::vector<std::vector<uint32_t>> result(rowsA, std::vector<uint32_t>(colsB));
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                result[i][j] = h_C[i * colsB + j];
            }
        }

        // Ожидание завершения всех операций
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        // Освобождение памяти
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;

        return result;
    }

    std::vector<std::vector<uint32_t>> mul_transpose(Matrix& matrix) {
        std::vector<std::vector<uint32_t>> matrix1 = this->readFile();
        std::vector<std::vector<uint32_t>> matrix2 = matrix.transpose();


        if (matrix1[0].size() != matrix2.size()) {
            throw std::runtime_error("Matrix dimensions are not compatible for multiplication");
        }


        std::vector<std::vector<uint32_t>> res(matrix1.size(), std::vector<uint32_t>(matrix2[0].size(), 0));

        for (size_t i = 0; i < matrix1.size(); i++) {
            for (size_t j = 0; j < matrix2[0].size(); j++) {
                for (size_t k = 0; k < matrix2.size(); k++) {
                    res[i][j] += matrix1[i][k] * matrix2[j][k];
                }
            }
        }
        return res;
    }
    std::vector<std::vector<uint32_t>> mul(Matrix& matrix) {
        std::vector<std::vector<uint32_t>> matrix1 = this->readFile();
        std::vector<std::vector<uint32_t>> matrix2 = matrix.readFile();


        if (matrix1[0].size() != matrix2.size()) {
            throw std::runtime_error("Matrix dimensions are not compatible for multiplication");
        }


        std::vector<std::vector<uint32_t>> res(matrix1.size(), std::vector<uint32_t>(matrix2[0].size(), 0));

        for (size_t i = 0; i < matrix1.size(); i++) {
            for (size_t j = 0; j < matrix2[0].size(); j++) {
                for (size_t k = 0; k < matrix2.size(); k++) {
                    res[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return res;
    }
};

int main() {
    try {
        checkCudaDevices();
        std::wofstream file_time("time_stats.txt", std::ios::ate);
        for (size_t XY = 100; XY < 2001; XY += 100)
        {
            Matrix matrix1("../../matrix_" + std::to_string(XY) + "on" + std::to_string(XY) + "_first.txt");
            Matrix matrix2("../../matrix_" + std::to_string(XY) + "on" + std::to_string(XY) + "_second.txt");


            // Тестирование CUDA умножения
            auto start_cuda = std::chrono::steady_clock::now();
            auto data_res_cuda = matrix1.mul_cuda(matrix2);
            auto end_cuda = std::chrono::steady_clock::now();
            auto time_cuda_ms = int(std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda - start_cuda).count());
            
            std::wstring str(std::to_wstring(XY) + L"on" + std::to_wstring(XY) + L"_time: " + std::to_wstring(time_cuda_ms));
            file_time.write(str.c_str(), str.size());
            file_time.write(L";\n", 3);
            file_time.flush();
            write_matrix(data_res_cuda, "../../matrix_" + std::to_string(XY) + "on" + std::to_string(XY) + "_res.txt");
            std::wcout << std::to_wstring(XY) + L"on" + std::to_wstring(XY) + L": time = " << time_cuda_ms << std::endl;
        }
        file_time.close();

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
