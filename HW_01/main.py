from __future__ import annotations

from numbers import Number
from typing import List, Tuple
from math import log, ceil

def gauss_matrix_mult(A: Matrix, B: Matrix) -> Matrix:
    ''' Multiply two matrices by using Gauss's algorithm

    Parameters
    ----------
    A: Matrix
        The first matrix to be multiplied
    B: Matrix
        The second matrix to be multiplied

    Returns
    -------
    Matrix
        The row-column multiplication of the matrices passed as parameters

    Raises
    ------
    ValueError
        If the number of columns of `A` is different from the number of
        rows of `B`
    '''
    
    if A.num_of_cols != B.num_of_rows:
        raise ValueError('The two matrices cannot be multiplied')
        
    result = [[0 for col in range(B.num_of_cols)]
              for row in range(A.num_of_rows)] # list of lists that represents final matrix
                
    for row in range(A.num_of_rows):
        for col in range(B.num_of_cols):
            value = 0
            for k in range(A.num_of_cols):
                value += A[row][k]*B[k][col]
            # in value we have the value which must be stored in position [row,col] in final matrix
            result[row][col] = value
    
    return Matrix(result, clone_matrix=False) 

    
def get_matrix_quadrants(A: Matrix) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    A11 = A.submatrix(0, A.num_of_rows//2, 0, A.num_of_cols//2)
    A12 = A.submatrix(0, A.num_of_rows//2, A.num_of_cols//2, A.num_of_cols//2)
    A21 = A.submatrix(A.num_of_rows//2, A.num_of_rows//2, 0, A.num_of_cols//2)
    A22 = A.submatrix(A.num_of_rows//2, A.num_of_rows//2, A.num_of_cols//2, A.num_of_cols//2)
    
    return A11, A12, A21, A22  

def strassen_matrix_mult(A: Matrix, B: Matrix) -> Matrix:
    ''' Standard version of Strassen's Algorithm
    -----
        The algorithm works for any pair of squared matrices of dimension 2^n x 2^n
    '''
    # Base case of Strassen algorithm
    if (max(A.num_of_rows, B.num_of_cols, A.num_of_cols) < 32):
        return gauss_matrix_mult(A,B)
    
    next_power_two = lambda x: 2 ** int(ceil(log(x, 2)))
    n_A = next_power_two(A.num_of_rows)
    m_A = next_power_two(A.num_of_cols)
    n_B = next_power_two(B.num_of_rows)
    m_B = next_power_two(B.num_of_cols)
    
    A_S = [[0 for j in range(m_A)] for i in range(n_A)]
    B_S = [[0 for j in range(m_B)] for i in range(n_B)]
    
    A_S = Matrix(A_S, clone_matrix=False)
    B_S = Matrix(B_S, clone_matrix=False)
    
    A_S.assign_submatrix(0,0,A)
    B_S.assign_submatrix(0,0,B)
    
    # Recursive step
    A11, A12, A21, A22 = get_matrix_quadrants(A_S)
    B11, B12, B21, B22 = get_matrix_quadrants(B_S)

    S1 = B12 - B22
    S2 = A11 + A12
    S3 = A21 + A22
    S4 = B21 - B11
    S5 = A11 + A22
    S6 = B11 + B22
    S7 = A12 - A22
    S8 = B21 + B22
    S9 = A11 - A21
    S10 = B11 + B12
    
    # Recursive calls
    P1 = strassen_matrix_mult(A11, S1)
    P2 = strassen_matrix_mult(S2, B22)
    P3 = strassen_matrix_mult(S3, B11)
    P4 = strassen_matrix_mult(A22, S4)
    P5 = strassen_matrix_mult(S5, S6)
    P6 = strassen_matrix_mult(S7, S8)
    P7 = strassen_matrix_mult(S9, S10)
    
    # Second bunch of sums Theta(n^2)
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7

    # Build the resulting matrix
    C = [[0 for col in range(B_S.num_of_cols)] for row in range(A_S.num_of_rows)]
    
    C = Matrix(C, clone_matrix=False)
    
    # Use the assign_submatrix method in the Matrix class to copy Cij
    C.assign_submatrix(0, 0, C11)
    C.assign_submatrix(0, C.num_of_cols//2, C12)
    C.assign_submatrix(C.num_of_rows//2, 0, C21)
    C.assign_submatrix(C.num_of_rows//2, C.num_of_cols//2, C22)
    
    return C

def strassen_matrix_mult_gen(A: Matrix, B: Matrix) -> Matrix:
    ''' General version of the Strassen's Algorithm
    
    This version works for any pair of matrices that can be multiplied
    -----
    The algorithm works for rectangular matrices
    The idea is to use dynamic padding at each recursion to make each size even    
    '''
    # Base case of Strassen algorithm
    if (max(A.num_of_rows, B.num_of_cols, A.num_of_cols) < 32 or
        min(A.num_of_rows, B.num_of_cols, A.num_of_cols) < 2):
        return gauss_matrix_mult(A,B)
    
    even_size = lambda x: 2 * int(ceil(x/2))
    n_Ae = even_size(A.num_of_rows) # updated num of rows A = num of rows of A_S
    m_Ae = even_size(A.num_of_cols) # updated num of cols A = num of cols of A_S
    n_Be = even_size(B.num_of_rows) # updated num of rows B = num of rows of B_S
    m_Be = even_size(B.num_of_cols) # updated num of cold B = num of rows of B_S
    
    A_S = [[0 for j in range(m_Ae)] for i in range(n_Ae)]
    B_S = [[0 for j in range(m_Be)] for i in range(n_Be)]
    
    A_S = Matrix(A_S, clone_matrix=False)
    B_S = Matrix(B_S, clone_matrix=False)
    
    A_S.assign_submatrix(0,0,A)
    B_S.assign_submatrix(0,0,B)
    
    # Recursive step
    A11, A12, A21, A22 = get_matrix_quadrants(A_S)
    B11, B12, B21, B22 = get_matrix_quadrants(B_S)
    
    S1 = B12 - B22
    S2 = A11 + A12
    S3 = A21 + A22
    S4 = B21 - B11
    S5 = A11 + A22
    S6 = B11 + B22
    S7 = A12 - A22
    S8 = B21 + B22
    S9 = A11 - A21
    S10 = B11 + B12
    
    # Recursive calls
    P1 = strassen_matrix_mult_gen(A11, S1)
    P2 = strassen_matrix_mult_gen(S2, B22)
    P3 = strassen_matrix_mult_gen(S3, B11)
    P4 = strassen_matrix_mult_gen(A22, S4)
    P5 = strassen_matrix_mult_gen(S5, S6)
    P6 = strassen_matrix_mult_gen(S7, S8)
    P7 = strassen_matrix_mult_gen(S9, S10)
    
    # Second bunch of sums Theta(n^2)
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7

    # Build the resulting matrix
    C = [[0 for col in range(m_Be)] for row in range(n_Ae)]    
    C = Matrix(C, clone_matrix=False)

    # Use the assign_submatrix method in the Matrix class to copy Cij
    C.assign_submatrix(0, 0, C11)
    C.assign_submatrix(0, C.num_of_cols//2, C12)
    C.assign_submatrix(C.num_of_rows//2, 0, C21)
    C.assign_submatrix(C.num_of_rows//2, C.num_of_cols//2, C22)      
    
    # Get rid of the zoro-valued rows and columns added with padding   
    C = [[C[i][j] for j in range(B.num_of_cols)] for i in range(A.num_of_rows)]
    C = Matrix(C,clone_matrix=False)
    
    return C

def strassen_matrix_mult_gen_opt(A: Matrix, B: Matrix) -> Matrix:
    ''' Memory-Optimized version of the General Strassen's Algorithm

    The idea is to reduce the number of auxiliary matrices by:
        - reusing the same memory space for each of the Pi matrices, by allocating a
          unique matrix and assign the partial sums to C as soon as they
          are available (i.e., before the next recursive call)
    '''
    # Base case of Strassen algorithm
    if (max(A.num_of_rows, B.num_of_cols, A.num_of_cols) < 32 or
        min(A.num_of_rows, B.num_of_cols, A.num_of_cols) < 2):
        return gauss_matrix_mult(A,B)
    
    even_size = lambda x: 2 * int(ceil(x/2))
    n_Ae = even_size(A.num_of_rows) # updated num of rows A = num of rows of A_S
    m_Ae = even_size(A.num_of_cols) # updated num of cols A = num of cols of A_S
    n_Be = even_size(B.num_of_rows) # updated num of rows B = num of rows of B_S
    m_Be = even_size(B.num_of_cols) # updated num of cold B = num of rows of B_S
    
    # Extract quadrants from padded matrix A
    Aux = [[0 for j in range(m_Ae)] for i in range(n_Ae)]
    Aux = Matrix(Aux, clone_matrix=False)
    Aux.assign_submatrix(0,0,A)    
    A11, A12, A21, A22 = get_matrix_quadrants(Aux)
    
    # Extract quadrants from padded matrix B
    Aux = [[0 for j in range(m_Be)] for i in range(n_Be)]
    Aux = Matrix(Aux, clone_matrix=False)  
    Aux.assign_submatrix(0,0,B)
    B11, B12, B21, B22 = get_matrix_quadrants(Aux)
    
    # Allocate the resulting matrix
    C = [[0 for col in range(m_Be)] for row in range(n_Ae)]    
    C = Matrix(C, clone_matrix=False)
    
    n_C = C.num_of_rows
    m_C = C.num_of_cols
    
    # Strassen Algorithm
    Aux = strassen_matrix_mult_gen_opt(A11, B12 - B22) # P1
    C.assign_submatrix(0, m_C//2, Aux) # C12 = P1 (C12 = P1 + P2) 
    C.assign_submatrix(n_C//2, m_C//2, Aux) # C22 = P1 (C22 = P5 + P1 - P3 - P7)
    
    Aux = strassen_matrix_mult_gen_opt(A11 + A12, B22) # P2
    C.assign_submatrix(0, 0, (-1) * Aux) # C11 = - P2 (C11 = P5 + P4 - P2 + P6)
    C.add_submatrix(0, m_C//2, Aux) # C12 = P1 + P2 ----------------------------------------------------------- C12 OK    
    
    Aux = strassen_matrix_mult_gen_opt(A21 + A22, B11) # P3
    C.assign_submatrix(n_C//2, 0, Aux) # C21 = P3 (C21 = P3 + P4)
    C.add_submatrix(n_C//2, m_C//2, (-1) * Aux) # C22 = P1 - P3 (C22 = P5 + P1 - P3 - P7)
    
    Aux = strassen_matrix_mult_gen_opt(A22, B21 - B11) # P4
    C.add_submatrix(0, 0, Aux) # C11 = P4 - P2 (C11 = P5 + P4 - P2 + P6)
    C.add_submatrix(n_C//2, 0, Aux) # C21 = P3 + P4 ----------------------------------------------------------- C21 OK
    
    Aux = strassen_matrix_mult_gen_opt(A11 + A22, B11 + B22) # P5
    C.add_submatrix(0, 0, Aux) # C11 = P5 + P4 - P2 (C11 = P5 + P4 - P2 + P6)
    C.add_submatrix(n_C//2, m_C//2, Aux)  # C22 = P5 + P1 - P3 (C22 = P5 + P1 - P3 - P7)
    
    Aux = strassen_matrix_mult_gen_opt(A12 - A22, B21 + B22) # P6
    C.add_submatrix(0, 0, Aux) # C11 = P5 + P4 - P2 + P6 ------------------------------------------------------ C11 OK
    
    Aux = strassen_matrix_mult_gen_opt(A11 - A21, B11 + B12) # P7
    C.add_submatrix(n_C//2, m_C//2, (-1) * Aux)  # C22 = P5 + P1 - P3 - P7 ------------------------------------ C22 OK

    # Get rid of the zoro-valued rows and columns added with padding    
    C = [[C[i][j] for j in range(B.num_of_cols)] for i in range(A.num_of_rows)]
    C = Matrix(C,clone_matrix=False)
    
    return C

class Matrix(object):
    ''' A simple naive matrix class

    Members
    -------
    _A: List[List[Number]]
        The list of rows that store all the matrix values

    Parameters
    ----------
    A: List[List[Number]]
        The list of rows that store all the matrix values
    clone_matrix: Optional[bool]
        A flag to require a full copy of `A`'s data structure.

    Raises
    ------
    ValueError
        If there are two lists having a different number of values
    '''
    def __init__(self, A: List[List[Number]], clone_matrix: bool = True):
        num_of_cols = None

        for i, row in enumerate(A):
            if num_of_cols is not None:
                if num_of_cols != len(row):
                    raise ValueError('This is not a matrix')
            else:
                num_of_cols = len(row)

        if clone_matrix:
            self._A = [[value for value in row] for row in A]
        else:
            self._A = A

    @property
    def num_of_rows(self) -> int:
        return len(self._A)

    @property
    def num_of_cols(self) -> int:
        if len(self._A) == 0:
            return 0

        return len(self._A[0])

    def copy(self):
        A = [[value for value in row] for row in self._A]

        return Matrix(A, clone_matrix=False)

    def __getitem__(self, y: int):
        ''' Return one of the rows

        Parameters
        ----------
        y: int
            the index of the rows to be returned

        Returns
        -------
        List[Number]
            The `y`-th row of the matrix
        '''
        return self._A[y]

    def __iadd__(self, A: Matrix) -> Matrix:
        ''' Sum a matrix to this matrix and update it

        Parameters
        ----------
        A: Matrix
            The matrix to be summed up

        Returns
        -------
        Matrix
            The matrix corresponding to the sum between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''

        if (self.num_of_cols != A.num_of_cols or
                self.num_of_rows != A.num_of_rows):
            raise ValueError('The two matrices have different sizes')

        for y in range(self.num_of_rows):
            for x in range(self.num_of_cols):
                self[y][x] += A[y][x]

        return self

    def __add__(self, A: Matrix) -> Matrix:
        ''' Sum a matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix to be summed up

        Returns
        -------
        Matrix
            The matrix corresponding to the sum between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''
        res = self.copy()

        res += A

        return res

    def __isub__(self, A: Matrix) -> Matrix:
        ''' Subtract a matrix to this matrix and update it

        Parameters
        ----------
        A: Matrix
            The matrix to be subtracted up

        Returns
        -------
        Matrix
            The matrix corresponding to the subtraction between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''

        if (self.num_of_cols != A.num_of_cols or
                self.num_of_rows != A.num_of_rows):
            raise ValueError('The two matrices have different sizes')

        for y in range(self.num_of_rows):
            for x in range(self.num_of_cols):
                self[y][x] -= A[y][x]

        return self

    def __sub__(self, A: Matrix) -> Matrix:
        ''' Subtract a matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix to be subtracted up

        Returns
        -------
        Matrix
            The matrix corresponding to the subtraction between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''
        res = self.copy()

        res -= A

        return res

    def __mul__(self, A: Matrix) -> Matrix:
        ''' Multiply one matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix which multiplies this matrix

        Returns
        -------
        Matrix
            The row-column multiplication between this matrix and that passed
            as parameter

        Raises
        ------
        ValueError
            If the number of columns of this matrix is different from the
            number of rows of `A`
        '''
        return gauss_matrix_mult(self, A)

    def __rmul__(self, value: Number) -> Matrix:
        ''' Multiply one matrix by a numeric value

        Parameters
        ----------
        value: Number
            The numeric value which multiplies this matrix

        Returns
        -------
        Matrix
            The multiplication between `value` and this matrix

        Raises
        ------
        ValueError
            If `value` is not a number
        '''

        if not isinstance(value, Number):
            raise ValueError('{} is not a number'.format(value))

        return Matrix([[value*elem for elem in row] for row in self._A],
                      clone_matrix=False)

    def submatrix(self, from_row: int, num_of_rows: int,
                  from_col: int, num_of_cols: int) -> Matrix:
        ''' Return a submatrix of this matrix

        Parameters
        ----------
        from_row: int
            The first row to be included in the submatrix to be returned
        num_of_rows: int
            The number of rows to be included in the submatrix to be returned
        from_col: int
            The first col to be included in the submatrix to be returned
        num_of_cols: int
            The number of cols to be included in the submatrix to be returned

        Returns
        -------
        Matrix
            A submatrix of this matrix
        '''
        A = [row[from_col:from_col+num_of_cols]
             for row in self._A[from_row:from_row+num_of_rows]]

        return Matrix(A, clone_matrix=False)

    def assign_submatrix(self, from_row: int, from_col: int, A: Matrix):
        for y, row in enumerate(A):
            self_row = self[y + from_row]
            for x, value in enumerate(row):
                self_row[x + from_col] = value
    
    def add_submatrix(self, from_row: int, from_col: int, A: Matrix): 
        for y, row in enumerate(A):
            self_row = self[y + from_row]
            for x, value in enumerate(row):
                self_row[x + from_col] += value

    def __repr__(self):
        return '\n'.join('{}'.format(row) for row in self._A)
    
    def compare_matrix(self, A: Matrix, tol: float = 1e-10) -> None:
        ''' Compares this matrix to a given matrix
        
        Parameters
        ----------
        A: Matrix
            The matrix which is compared to this matrix
        tol: int
            The tolerance in the values comparison
        
        Returns
        -------
        None
        '''
        if (self.num_of_cols != A.num_of_cols or
                self.num_of_rows != A.num_of_rows):
            raise ValueError('The two matrices have different sizes')
        
        Equal = True
        for i in range(self.num_of_rows):
            for j in range(self.num_of_cols):
                if abs(self[i][j]-A[i][j]) > 1e-10:
                    Equal = False
          
        if Equal is True:
            stdout.write(f'\nThe two matrices are equal, tolerance = {tol}\n')
        else:
            stdout.write(f'\nThe two matrices are NOT equal, tolerance = {tol}\n')


class IdentityMatrix(Matrix):
    ''' A class for identity matrices

    Parameters
    ----------
    size: int
        The size of the identity matrix
    '''
    def __init__(self, size: int):
        A = [[1 if x == y else 0 for x in range(size)]
             for y in range(size)]

        super().__init__(A)

if __name__ == '__main__':

    from random import random, seed
    from sys import stdout
    from timeit import timeit
    seed(0) #to be able to debug the code
    
    ## Timing for matrices 2**n x 2**n
    stdout.write('\nTiming analysis for matrices 2**n x 2**n')
    stdout.write('\n\tG\tS G\tS G-M\n')
    for idx in range(1,10):
        size = 2**idx
        A = Matrix([[random() for j in range(size)] for i in range(size)])
        B = Matrix([[random() for j in range(size)] for i in range(size)])
        stdout.write(f'{size}')
        for funct in ['gauss_matrix_mult', 'strassen_matrix_mult_gen','strassen_matrix_mult_gen_opt']:
            T = timeit(f'{funct}(A,B)', globals=locals(), number=1) 
            stdout.write('\t{:.3f}'.format(T))
            stdout.flush()
        stdout.write('\n')
 
    ## Timing for the general case 
    # Consider the following cases:
    # 1. Square matrices but sizes not a power of 2
    # 2. Rectangular matrices
    n_A = [300, 351, 412]
    m_A = [300, 351, 327]
    n_B = m_A
    m_B = [300, 351, 283]
 
    stdout.write('\n\nTiming analysis for generic matrices (also rectangular)') 
    stdout.write('\n\tG\tS G\tS G-M\n')
    for idx in range(len(n_A)):
        A = Matrix([[random() for j in range(m_A[idx])] for i in range(n_A[idx])])
        B = Matrix([[random() for j in range(m_B[idx])] for i in range(n_B[idx])])
        stdout.write(f'{max(n_A[idx],m_A[idx],m_B[idx])}')
        for funct in ['gauss_matrix_mult', 'strassen_matrix_mult_gen','strassen_matrix_mult_gen_opt']:
            T = timeit(f'{funct}(A,B)', globals=locals(), number=1) 
            stdout.write('\t{:.3f}'.format(T))
            stdout.flush()
        stdout.write('\n')   

    ## Benchmark analysis
    stdout.write('\n\nLets perform some checks on the goodness of results...\n')
    A = Matrix([[random() for j in range(2**7)] for i in range(2**7)])
    B = Matrix([[random() for j in range(2**7)] for i in range(2**7)])
    C_G = gauss_matrix_mult(A, B)
    C_S = strassen_matrix_mult(A, B)
    stdout.write('\nComparison between Gauss Algorithm and Strassen Algorithm for matrices 2**n x 2**n')
    C_S.compare_matrix(C_G)    
    
    A = Matrix([[random() for j in range(m_A[-1])] for i in range(n_A[-1])])
    B = Matrix([[random() for j in range(m_B[-1])] for i in range(n_B[-1])])
    C_G = gauss_matrix_mult(A, B)
    C_S_G = strassen_matrix_mult_gen(A, B)  
    C_S_GM = strassen_matrix_mult_gen_opt(A, B)
    stdout.write('\nComparison between Gauss Algorithm and General Strassen Algorithm')
    C_S_G.compare_matrix(C_G)
    stdout.write('\nComparison between Gauss Algorithm and General Strassen Algorithm Memory-Optimized')
    C_S_GM.compare_matrix(C_G)
   
