import numpy as np

def read_matrix_from_file(filename):
    return np.loadtxt(filename, delimiter=',', dtype=int)

def print_matrix(matrix):
    for row in matrix:
        print(','.join(map(str, row)))

def compare_matrices(matrix1, matrix2):
    return np.array_equal(matrix1, matrix2)

def mul(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

if __name__ == "__main__":
    for i in range(100, 2001, 100):
        matrix1 = read_matrix_from_file(f'matrix_{i}on{i}_first.txt')
        matrix2 = read_matrix_from_file(f'matrix_{i}on{i}_second.txt')
        
        if matrix1.shape[1] != matrix2.shape[0]:
            print("Матрицы не могут быть перемножены из-за несовместимых размеров.")
        else:
            result = np.dot(matrix1, matrix2)
            # print("Вычисленный результат:")
            # print_matrix(result)
            
            expected_result = read_matrix_from_file(f'matrix_{i}on{i}_res.txt')
            if compare_matrices(result, expected_result):
                print(f"{i}on{i}: Вычисленный результат совпадает с ожидаемым результатом.")
            else:
                print(f"{i}on{i}: Вычисленный результат НЕ СОВПАДАЕТ с ожидаемым результатом.")
