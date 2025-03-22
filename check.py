import numpy as np

def read_matrix_from_file(filename):
    return np.loadtxt(filename, delimiter=',', dtype=int)

def print_matrix(matrix):
    for row in matrix:
        print(','.join(map(str, row)))

def compare_matrices(matrix1, matrix2):
    return np.array_equal(matrix1, matrix2)

if __name__ == "__main__":
    matrix1 = read_matrix_from_file('matrix_1000on1000_first.txt')
    matrix2 = read_matrix_from_file('matrix_1000on1000_second.txt')
    
    if matrix1.shape[1] != matrix2.shape[0]:
        print("Матрицы не могут быть перемножены из-за несовместимых размеров.")
    else:
        result = np.dot(matrix1, matrix2)
        # print("Вычисленный результат:")
        # print_matrix(result)
        
        expected_result = read_matrix_from_file('matrix_1000on1000_res.txt')
        if compare_matrices(result, expected_result):
            print("Вычисленный результат совпадает с ожидаемым результатом.")
        else:
            print("Вычисленный результат не совпадает с ожидаемым результатом.")
