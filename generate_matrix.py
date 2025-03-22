import argparse
import random
import numpy as np

def generate_matrix_file(filename: str, rows: int, cols: int):
    """
    Generates a matrix file with given dimensions and writes it to a file.
    Each row contains comma-separated integers.
    """
    with open(filename, "w") as f:
        for _ in range(rows):
            row = [str(random.randint(0, 1000)) for _ in range(cols)]
            f.write(",".join(row) + "\n")

if __name__ == "__main__":
    # Создаем парсер
    parser = argparse.ArgumentParser()

    # Добавляем аргументы
    parser.add_argument("size", type=int, help="Размер стороны квадратной матрицы", default=100)

    # Парсим аргументы
    args = parser.parse_args()

    generate_matrix_file(f"matrix_{args.size}on{args.size}.txt", args.size, args.size)
    


