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
    for i in range(2500, 10001, 500):
        generate_matrix_file(f"matrix_{i}on{i}_first.txt", i, i)
    for i in range(2500, 10001, 500):
        generate_matrix_file(f"matrix_{i}on{i}_second.txt", i, i)
    


