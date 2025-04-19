# import generate_matrix
import check

if __name__ == "__main__":

    for i in range(100, 2001, 100):
        matrix1 = check.read_matrix_from_file('matrix_{i}on{i}_first.txt', )
        matrix2 = check.read_matrix_from_file('matrix_{i}on{i}_second.txt.txt')
        if matrix1.shape[1] != matrix2.shape[0]:
            print("Матрицы не могут быть перемножены из-за несовместимых размеров.")
        else:
            res = check.dot()
            matrix2_res = check.read_matrix_from_file('matrix_{i}on{i}_res.txt')
    
    
    