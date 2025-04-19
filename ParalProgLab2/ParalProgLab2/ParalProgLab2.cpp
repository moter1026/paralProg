#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <omp.h>

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
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < matrix.size(); ++i) {
            for (int j = 0; j < matrix[i].size(); ++j) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }

    std::vector<std::vector<uint32_t>> mul_transpose(Matrix& matrix) {
        std::vector<std::vector<uint32_t>> matrix1 = this->readFile();
        std::vector<std::vector<uint32_t>> matrix2 = matrix.transpose();


        if (matrix1[0].size() != matrix2.size()) {
            throw std::runtime_error("Matrix dimensions are not compatible for multiplication");
        }


        std::vector<std::vector<uint32_t>> res(matrix1.size(), std::vector<uint32_t>(matrix2[0].size(), 0));

#pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < matrix1.size(); i++) {
            for (int j = 0; j < matrix2[0].size(); j++) {
                for (int k = 0; k < matrix2.size(); k++) {
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

#pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < matrix1.size(); i++) {
            for (int j = 0; j < matrix2[0].size(); j++) {
                for (int k = 0; k < matrix2.size(); k++) {
                    res[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return res;
    }
};

int main() {
    setlocale(LC_ALL, "ru");
    try {
        std::ofstream file_time("time_stats.txt", std::ios::ate);
        std::cout << "START" << std::endl;
#ifdef _OPENMP
#pragma omp parallel
    #pragma omp critical
        {
            if (omp_get_thread_num() != 0) {
                std::cout << "OpenMP Version: " << _OPENMP / 100 << "." << _OPENMP % 100 << std::endl;
                std::cout << "Процессоров: " << omp_get_num_procs()
                    << ", Потоков: " << omp_get_num_threads() << std::endl;
            }
        }
        omp_set_num_threads(omp_get_max_threads());
#else
        std::cout << "Последовательная версия (OpenMP не активен!)" << std::endl << std::endl;
#endif

        for (size_t XY = 100; XY < 2001; XY += 100)
        {
            Matrix matrix1("../../matrix_" + std::to_string(XY) + "on" + std::to_string(XY) + "_first.txt");
            Matrix matrix2("../../matrix_" + std::to_string(XY) + "on" + std::to_string(XY) + "_second.txt");


            auto start = std::chrono::steady_clock::now();
            //auto data_res = matrix1.mul_transpose(matrix2);
            auto data_res = matrix1.mul(matrix2);
            auto end = std::chrono::steady_clock::now();

            auto time_ms = int(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
            std::string str(std::to_string(XY) + "on" + std::to_string(XY) + "_time: " + std::to_string(time_ms));
            file_time.write(str.c_str(), str.size());
            file_time.write(";\n", 3);
            file_time.flush();
            write_matrix(data_res, "../../matrix_" + std::to_string(XY) + "on" + std::to_string(XY) + "_res.txt");
            std::cout << std::to_string(XY) + "on" + std::to_string(XY) + ": time = " << time_ms << std::endl;
        }
        file_time.close();
        std::cout << "THE END" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
