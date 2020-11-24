import numpy as np
from collections.abc import Iterable, Iterator


class Matrix(Iterable):
    def __init__(self, mat):
        self.matrix = np.array(mat)

    def addition(self, matrix):
        try:
            pass
        except:
            print("NOPe")
        return self.matrix + matrix.matrix

    def subtraction(self, matrix):
        return self.matrix - matrix.matrix

    def multiply(self, matrix):
        return self.matrix.dot(matrix.matrix)  # self.matrix * matrix.matrix

    def transpose(self):
        return self.matrix.transpose()

    def inverse_matrix(self): # A * A^-1 n != m
        try:
            return np.linalg.inv(self.matrix)
        except:
            print("Входная матрица не является квадратной или вычисление обратной матрицы невозможно")

    def norm_matrix(self):
        return np.linalg.norm(self.matrix)

    def linear_equation(self, b):
        """ где self.matrix - коефициенты при x0, x1, x2
            b - решение уравнения
            Пример:
            x0 + 2x1 - 3x2 = 4,
            2x0 + x1 + 2x2 = 3,
            3x0 - 2x2 - x2 = 9.
            Тогда [[1 2 -3]
                   [2 1 2]
                   [3 -2 -1]]
            b = [4, 3, 9]"""
        try:
            return np.linalg.solve(self.matrix, b)
        except:
            print("""Данная функция вычисляет значение неизвестных только для квадратных, 
                    невырожденных матриц с полным рангом, т.е.
                     только если матрица A размером {m, m} имеет ранг равный m""")

    def eigh(self):
        return np.linalg.eigh(self.matrix)

    def generator(self, m, n):
        k = 0
        l = 0
        ''' k - starting row index
            m - ending row index
            l - starting column index
            n - ending column index
            i - iterator '''

        while (k < m and l < n):

            # Print the first row from
            # the remaining rows
            for i in range(l, n):                     ### 1 2 -3
                yield self.matrix[k][i]

            k += 1

            # Print the last column from
            # the remaining columns
            for i in range(k, m):           ##
                yield self.matrix[i][n - 1]

            n -= 1

            # Print the last row from
            # the remaining rows
            if (k < m):

                for i in range(n - 1, (l - 1), -1):
                    yield self.matrix[m - 1][i]

                m -= 1

            # Print the first column from
            # the remaining columns
            if (l < n):
                for i in range(m - 1, k - 1, -1):
                    yield self.matrix[i][l]

                l += 1


    def __iter__(self):
        return MatrixIterator(self.matrix, len(self.matrix[0]))

    def display_matrix(self):                                    #
        print(self.matrix)


class MatrixIterator(Iterator):
    def __init__(self, collection, length):
        self.matrix = collection
        self.length = length
        self.positionI = 0
        self.positionJ = 0

    def __next__(self):
        try:
            value = self.matrix[self.positionI][self.positionJ]
            if (self.positionJ + 1) >= self.length:
                self.positionI += 1
                self.positionJ = -1
            self.positionJ += 1
        except IndexError:
            raise StopIteration()

        return value


def main():
     print("Start APP")
     while True:
         print("\nMods: Exit, Addition, Subtraction, Multiply\nTranspose, Inverse_matrix, Norm_matrix, Eigh\nLinear_equation, Display_matrix, Display_spiral_matrix\n")
         action = input("Enter a mode :> ").lower()
         if action == "exit":
             break
         elif action == "addition":
             first_matrix, second_matrix = displayCreateMatrix()
             print(first_matrix.addition(second_matrix))
         elif action == "subtraction":
             first_matrix, second_matrix = displayCreateMatrix()
             print(first_matrix.subtraction(second_matrix))
         elif action == "multiply":
             first_matrix, second_matrix = displayCreateMatrix()
             print(first_matrix.multiply(second_matrix))
         elif action == "transpose":
             first_matrix = displayCrateSingleMatrix()
             print(first_matrix.transpose())
         elif action == "inverse_matrix":
             first_matrix = displayCrateSingleMatrix()
             print(first_matrix.inverse_matrix())
         elif action == "norm_matrix":
             first_matrix = displayCrateSingleMatrix()
             print(first_matrix.inverse_matrix())
         elif action == "eigh":
             first_matrix = displayCrateSingleMatrix()
             print(first_matrix.eigh())
         elif action == "linear_equation":
             first_matrix = displayCrateSingleMatrix()
             print("Please input solution: ")
             solArray = []
             for i in range(len(first_matrix.matrix)):
                 solArray.append(float(input("Enter a value :> ")))
             print(first_matrix.linear_equation(solArray))
         elif action == "display_matrix":
             first_matrix = displayCrateSingleMatrix()
             for i in first_matrix:
                 print(i)
         elif action == "display_spiral_matrix":
             first_matrix = displayCrateSingleMatrix()
             iterator = first_matrix.generator(len(first_matrix.matrix), len(first_matrix.matrix[0]))
             for i in iterator:
                 print(i, end=" ")


### helper functions

def displayCrateSingleMatrix():
    print("\nEnter first matrix: ")
    first_matrix = Matrix(input_matrix())
    return first_matrix


def displayCreateMatrix():
    print("\nEnter first matrix: ")
    first_matrix = Matrix(input_matrix())
    print("Enter second matrix: ")
    second_matrix = Matrix(input_matrix())
    return (first_matrix, second_matrix)


def input_matrix():
    n = int(input("Enter size n (rows) :> "))
    m = int(input("Enter size m (columns) :> "))
    matrix = [[0 for n in range(m)] for nn in range(n)]
    for i in range(n):
        for j in range(m):
            value = float(input("Enter a value :> "))
            matrix[i][j] = value

    return matrix


if __name__ == "__main__":
    main()