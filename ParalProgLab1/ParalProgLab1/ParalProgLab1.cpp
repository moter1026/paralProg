#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>

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

    std::vector<std::vector<uint32_t>> readFile(){
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

    std::vector<std::vector<uint32_t>> mul(Matrix& matrix){
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
        Matrix matrix1("../../matrix_1000on1000_first.txt");
        Matrix matrix2("../../matrix_1000on1000_second.txt");

        auto data_res = matrix1.mul(matrix2);

        write_matrix(data_res, "../../matrix_1000on1000_res.txt");
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
