#include "mpi.h"
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <chrono>

void write_matrix(std::vector<std::vector<uint32_t> >& matrix, std::string nameFile) {
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

    std::vector<std::vector<uint32_t> > readFile(){
        std::vector<std::vector<uint32_t> > matrix;
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
    std::vector<std::vector<uint32_t> > transpose() {
        std::vector<std::vector<uint32_t> > matrix = readFile();

        // Если матрица пустая, возвращаем пустую матрицу
        if (matrix.empty() || matrix[0].empty()) {
            return {};
        }

        // Создаем новую матрицу с перевернутыми размерами
        std::vector<std::vector<uint32_t> > transposed(matrix[0].size(), std::vector<uint32_t>(matrix.size()));

        // Заполняем транспонированную матрицу
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }

    std::vector<std::vector<uint32_t> > mul_transpose(Matrix& matrix){
        std::vector<std::vector<uint32_t> > matrix1 = this->readFile();
        std::vector<std::vector<uint32_t> > matrix2 = matrix.transpose();


        if (matrix1[0].size() != matrix2.size()) {
            throw std::runtime_error("Matrix dimensions are not compatible for multiplication");
        }


        std::vector<std::vector<uint32_t> > res(matrix1.size(), std::vector<uint32_t>(matrix2[0].size(), 0));

        for (size_t i = 0; i < matrix1.size(); i++) {
            for (size_t j = 0; j < matrix2[0].size(); j++) {
                for (size_t k = 0; k < matrix2.size(); k++) {
                    res[i][j] += matrix1[i][k] * matrix2[j][k];
                }
            }
        }
        return res;
    }
    std::vector<std::vector<uint32_t> > mul(Matrix& matrix) {
        std::vector<std::vector<uint32_t> > matrix1 = this->readFile();
        std::vector<std::vector<uint32_t> > matrix2 = matrix.readFile();


        if (matrix1[0].size() != matrix2.size()) {
            throw std::runtime_error("Matrix dimensions are not compatible for multiplication");
        }


        std::vector<std::vector<uint32_t> > res(matrix1.size(), std::vector<uint32_t>(matrix2[0].size(), 0));

        for (size_t i = 0; i < matrix1.size(); i++) {
            for (size_t j = 0; j < matrix2[0].size(); j++) {
                for (size_t k = 0; k < matrix2.size(); k++) {
                    res[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return res;
    }

    static std::vector<std::vector<uint32_t> > mul_MPI(Matrix& matrix_A, Matrix& matrix_B, int rank, int size) {
        std::vector<std::vector<uint32_t> > matrix1, matrix2;
        int N, M, P;

        try {
            // Только root процесс читает и подготавливает данные
            if (rank == 0) {
                matrix1 = matrix_A.readFile();
                matrix2 = matrix_B.transpose();

                if (matrix1[0].size() != matrix2.size()) {
                    throw std::runtime_error("Matrix dimensions are not compatible for multiplication");
                }

                N = matrix1.size();
                M = matrix1[0].size();
                P = matrix2[0].size();
            }

            // Рассылка размеров матриц всем процессам
            MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Выделение памяти под данные
            std::vector<uint32_t> flattened_A(N * M), flattened_B(M * P);

            // Только root заполняет данные
            if (rank == 0) {
                for (int i = 0; i < N; ++i)
                    std::copy(matrix1[i].begin(), matrix1[i].end(), flattened_A.begin() + i * M);

                for (int i = 0; i < M; ++i) // matrix2 уже транспонирована
                    std::copy(matrix2[i].begin(), matrix2[i].end(), flattened_B.begin() + i * P);
            }

            // Распределение матрицы A
            std::vector<int> sendcounts(size), displs(size);
            const int base_rows = N / size;
            const int extra_rows = N % size;

            int current_displ = 0;
            for (int i = 0; i < size; ++i) {
                const int rows = base_rows + (i < extra_rows);
                sendcounts[i] = rows * M;
                displs[i] = current_displ;
                current_displ += sendcounts[i];
            }

            // Локальные данные
            const int local_rows = sendcounts[rank] / M;
            std::vector<uint32_t> local_A(sendcounts[rank]);
            std::vector<uint32_t> local_B(M * P);

            // Распределение A и B
            MPI_Scatterv(
                flattened_A.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                local_A.data(), sendcounts[rank], MPI_UNSIGNED,
                0, MPI_COMM_WORLD
            );

            MPI_Bcast(flattened_B.data(), M * P, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

            // Локальное умножение с оптимизированным доступом к памяти
            std::vector<uint32_t> local_res(local_rows * P, 0);
            for (int i = 0; i < local_rows; ++i) {
                const uint32_t* a_row = &local_A[i * M];
                for (int j = 0; j < P; ++j) {
                    const uint32_t* b_col = &flattened_B[j * M];
                    for (int k = 0; k < M; ++k) {
                        local_res[i * P + j] += a_row[k] * b_col[k];
                    }
                }
            }

            // Сбор результатов
            std::vector<int> recvcounts(size), recvdispls(size);
            current_displ = 0;
            for (int i = 0; i < size; ++i) {
                const int rows = base_rows + (i < extra_rows);
                recvcounts[i] = rows * P;
                recvdispls[i] = current_displ;
                current_displ += recvcounts[i];
            }

            std::vector<uint32_t> global_result;
            if (rank == 0) global_result.resize(N * P);

            MPI_Gatherv(
                local_res.data(), local_rows * P, MPI_UNSIGNED,
                global_result.data(), recvcounts.data(), recvdispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD
            );

            // Формирование результата только на root
            std::vector<std::vector<uint32_t> > result;
            if (rank == 0) {
                result.resize(N);
                for (int i = 0; i < N; ++i) {
                    result[i].resize(P);
                    std::copy(
                        global_result.begin() + i * P,
                        global_result.begin() + (i + 1) * P,
                        result[i].begin()
                    );
                }
            }
            return result;

        }
        catch (const std::exception& e) {
            std::cerr << "Rank " << rank << " error: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return {};
        }
    }


};

int main(int  argc, char** argv)
{
    try {
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);


        std::wofstream file_time("time_stats.txt", std::ios::ate);
        for (size_t XY = 100; XY < 2001; XY+= 100)
        {
            Matrix matrix1("../../../matrix_" + std::to_string(XY) + "on" + std::to_string(XY) + "_first.txt");
            Matrix matrix2("../../../matrix_" + std::to_string(XY) + "on" + std::to_string(XY) + "_second.txt");


            auto start = std::chrono::steady_clock::now();
            auto data_res = Matrix::mul_MPI(matrix1, matrix2, rank, size);

            auto end = std::chrono::steady_clock::now();

            auto time_ms = int(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
            std::wstring str(std::to_wstring(XY) + L"on" + std::to_wstring(XY) + L"_time: " + std::to_wstring(time_ms));
            file_time.write(str.c_str(), str.size());
            file_time.write(L";\n", 3);
            file_time.flush();
            write_matrix(data_res, "../../../matrix_" + std::to_string(XY) + "on" + std::to_string(XY) + "_res.txt");
            if (rank == 0)
            {
                std::wcout << std::to_wstring(XY) + L"on" + std::to_wstring(XY) + L": time = " << time_ms << std::endl;
            }
        }
        file_time.close();
        MPI_Finalize();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
