#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h>

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


void write_matrix(std::vector<std::vector<uint32_t>>& matrix, std::string nameFile) {
    std::ofstream file_res(nameFile);
    if (!file_res.is_open()) {
        throw std::runtime_error("File not open");
    }

    for (size_t i = 0; i < matrix.size(); i++)
    {
        for (size_t j = 0; j < matrix[i].size(); j++)
        {
            file_res << matrix[i][j];
            if (j + 1 != matrix[i].size()) file_res << ",";
            else file_res << "\n";
        }
    }
    file_res.close();
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
        dim3 blockSize(16, 16);
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
