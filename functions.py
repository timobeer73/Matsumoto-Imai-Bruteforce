import numpy as np
from datetime import datetime
from typing import List, Tuple
from argparse import Namespace
from math import pow as mathPow


def readFile(args: Namespace) -> Tuple[List[str], List[str], int]:
    """
    Read and process an input file into 3 variables.

    Returns:
        Tuple[list, list, int]: A tuple containing the public key (list), 
                                cipher text (list), and relationsAmount (int) extracted 
                                from the file.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Processing file \'{args.filepath}\'')

    # Read the given file.
    with open(args.filepath, 'r') as file:
        text = file.read()

    # Remove blank spaces and linebreaks for easier processing.
    text = text.replace(' ', '').replace('\n', '')

    # Separating the text into its variables.
    try:
        publicKey = text.split('[')[1].split(']')[0].split(',')
        cipherText = text.split('[')[2].split(']')[0].split(',')
        relationsAmount = len(cipherText)
    except:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Unable to locate all parameters from the file {args.filepath}.\n'
              f'\t   Check for the right formatting.')
        exit(-1)

    return publicKey, cipherText, relationsAmount


def generatePlainText(args: Namespace, relationsAmount: int) -> np.ndarray:
    """
    Generate a 2D numpy array of random plain texts.

    Args:
        relationsAmount (int): The number of basic elements/special relations.

    Returns:
        numpy.ndarray: A 2D numpy array of shape (2 * relationsAmount, relationsAmount) 
                       containing random plain texts represented by 
                       binary values.
    """
    plainTextAmount = 2 * mathPow(relationsAmount, 2)

    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Generating {round(plainTextAmount)} plain texts')

    plainTextMatrix = np.zeros(shape=(1, relationsAmount), 
                               dtype=np.bool_)

    # Generate random plain texts until 2 * relationsAmount² rows were generated.
    while plainTextMatrix.shape[0] < plainTextAmount:
        plainTextMatrix = np.vstack((plainTextMatrix, 
                                     np.random.choice(a=np.array([True, False]), 
                                                      size=(1, relationsAmount))))

    return plainTextMatrix


def calculateCipherText(args: Namespace, publicKey: List[str], plainTextMatrix: np.ndarray) -> np.ndarray:
    """
    Calculate cipher texts using the public key and plain text matrix.

    Args:
        publicKey (List[str]): List of strings representing the public key with variable placeholders.
        plainTextMatrix (numpy.ndarray): 2D numpy array containing plain text values.

    Returns:
        numpy.ndarray: 2D numpy array containing calculated cipher text values.
    """
    arrayDimensions = plainTextMatrix.shape
    cipherTextMatrix = np.zeros(shape=arrayDimensions, 
                                dtype=np.bool_)

    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Calculating {arrayDimensions[0]} corresponding cipher texts')

    # Replace the variables x_n of the public key with the corresponding plain text values to calculate the cipher text
    for row in range(0, arrayDimensions[0]):
        for column, publicKeyRow in enumerate(publicKey):
            for variable in reversed(range(0, arrayDimensions[1])):
                publicKeyRow = publicKeyRow.replace(f'x_{variable + 1}', str(plainTextMatrix[row][variable]))
            cipherTextMatrix[row][column] = eval(publicKeyRow) % 2

    return cipherTextMatrix


def calculateMatrix(args: Namespace, plainTextMatrix: np.ndarray, cipherTextMatrix: np.ndarray) -> np.ndarray:
    """
    Calculate a matrix by performing logical AND operations between plain text and cipher text matrices.

    Args:
        plainTextMatrix (numpy.ndarray): A 2D numpy array containing plain text values.
        cipherTextMatrix (numpy.ndarray): A 2D numpy array containing cipher text values.

    Returns:
        numpy.ndarray: A 2D numpy array containing the result of logical AND operations between
                       corresponding elements of the input matrices.
    """
    matrixDimension = plainTextMatrix.shape[0], plainTextMatrix.shape[1] * cipherTextMatrix.shape[1]
    matrix = np.zeros(shape=matrixDimension, 
                      dtype=np.bool_)

    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Calculating matrix from plain and cipher text')

    # Logical AND every single column of a plain text row with every column of the cipher text
    for row in range(0, matrixDimension[0]):
        for plainTextColumn in range(0, plainTextMatrix.shape[1]):
            for cipherTextColumn in range(0, cipherTextMatrix.shape[1]):
                matrix[row][plainTextColumn * plainTextMatrix.shape[1] + cipherTextColumn] = \
                    bool(plainTextMatrix[row][plainTextColumn]) and bool(cipherTextMatrix[row][cipherTextColumn])

    return matrix


def gaussianElimination(args: Namespace, matrix: np.ndarray) -> np.ndarray:
    """
    Perform Gaussian elimination on a binary matrix to simplify and solve the system of equations.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing a binary matrix.

    Returns:
        numpy.ndarray: A 2D numpy array representing the simplified matrix after Gaussian elimination.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Starting gaussian elimination')
    
    solvedMatrix = np.zeros(shape=[0, matrix.shape[1]], 
                            dtype=np.bool_)
    for columnIndex in range(0, matrix.shape[1]):
        # Remove every duplicate and False/zero only rows
        matrix = np.unique(ar=matrix, 
                           axis=0)
        matrix = matrix[~np.all(matrix == False, 
                                axis=1)]
        
        if matrix.shape[0] > 1:
            # Move all rows with a True/one in the nth column to the top of the matrix
            matrix = np.flipud(matrix[matrix[:, columnIndex].argsort()])

            # Move the row with the XXX amount of Trues/ones to the top
            optimalRowSum = 0
            optimalRowIndex = 0
            for rowIndex, row in enumerate(matrix):
                if row[columnIndex] == True:
                    rowSum = np.count_nonzero(row)
                    if rowSum > optimalRowSum:
                        optimalRowSum = rowSum
                        optimalRowIndex = rowIndex
                else:
                    break
            if optimalRowIndex != 0:
                matrix[[0, optimalRowIndex]] = matrix[[optimalRowIndex, 0]]

            # Logical XOR the current pivot row with all followings rows, which contain a True/one in the nth column
            if matrix[0][columnIndex] == True:
                for rowIndex in range(1, matrix.shape[0]):
                    if matrix[rowIndex][columnIndex] == True:
                        matrix[rowIndex][:] = np.logical_xor(matrix[0][:], matrix[rowIndex][:])
                    else:
                        break
            else:
                continue
        
        if matrix.shape[0] > 0:  
            # Store the current pivot row in the output matrix and delete the same row in the input matrix
            solvedMatrix = np.vstack([solvedMatrix, matrix[0][:]])
            matrix = np.delete(arr=matrix, 
                               obj=0, 
                               axis=0)
        else:
            break
        
    return solvedMatrix


def getFreeVariables(args: Namespace, matrix: np.ndarray) -> List[int]:
    """
    Find and return the indices of free variables in the solved binary matrix.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing the simplified matrix after Gaussian elimination.

    Returns:
        List[int]: A list of integers representing the indices of free variables.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Searching for free variables')

    # Check if the ith column and row is True/one. If not add it to the free variables
    freeVariables = []
    offset = 0
    for i in range(0, matrix.shape[1]):
        try:
            if matrix[i - offset][i] == False:
                freeVariables.append(i)
                offset += 1
        except:
            break
    
    # Add additional free variables beyond the current matrix row size
    for i in range(matrix.shape[0] + offset, matrix.shape[1]):
        freeVariables.append(i)

    return freeVariables


def reduceMatrix(args: Namespace, matrix: np.ndarray, freeVariables: List[int]) -> np.ndarray:
    """
    Reduce a binary matrix by performing additional operations based on free variables.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing a binary matrix.
        freeVariables (List[int]): A list of integers representing the indices of free variables.

    Returns:
        numpy.ndarray: A 2D numpy array representing the reduced binary matrix.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Reducing matrix')

    for column in range(1, matrix.shape[1]):
        # If the column is not a free variable and not fully reduced
        if column not in freeVariables and np.sum(matrix.T[column]) > 1:
            currentPivotRow = 0
            # Iterate upwards through the rows
            for row in range(matrix.shape[0] - 1, -1, -1):
                value = matrix[row][column]
                # Find the pivot element and logical XOR every row above containing a True/one
                if value == True:
                    if currentPivotRow == 0:
                        currentPivotRow = row
                    else:
                        matrix[row][:] = np.logical_xor(matrix[row][:], matrix[currentPivotRow][:])

    matrix = matrix[~np.all(matrix == False, 
                            axis=1)]

    return matrix


def getBaseVectors(args: Namespace, matrix: np.ndarray, freeVariables: List[int]) -> List[np.ndarray]:
    """
    Find and return the base vectors from a binary matrix based on free variables.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing a binary matrix.
        freeVariables (List[int]): A list of integers representing the indices of free variables.

    Returns:
        List[numpy.ndarray]: A list of numpy arrays representing the base vectors.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Searching for base vectors')
    
    # Insert extra rows to extract complete vectors
    for variable in freeVariables:
        matrix = np.insert(arr=matrix, 
                           obj=variable,
                           values=np.zeros(shape=[1, matrix.shape[1]], 
                                           dtype=np.bool_), 
                           axis=0)
        matrix[variable][variable] = 1

    # Save all columns which represent free variables
    baseVectors = []
    matrix = matrix.T
    for variable in reversed(freeVariables):
        baseVectors.append(matrix[:][variable])

    return baseVectors


def calculateRelationsMatrix(args: Namespace, baseVectors: List[np.ndarray], relationsAmount: int, cipherText: List[str]) -> np.ndarray:
    """
    Calculate a relations matrix based on base vectors and a cipher text matrix.

    Args:
        baseVectors (List[numpy.ndarray]): A list of numpy arrays representing base vectors.
        relationsAmount (int): The number of relations to calculate.
        cipherText (List[str]): An array representing a cipher text.

    Returns:
        numpy.ndarray: A 2D numpy array representing the relations matrix.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Calculating relations matrix')
    
    relationsMatrix = np.zeros(shape=[0, relationsAmount], 
                               dtype=np.bool_)

    # Logical AND every position of the ciphertext with 'relationsAmount'-large parts of the base vectors
    for vector in baseVectors:
        relation = np.zeros(shape=[1, relationsAmount], 
                            dtype=np.bool_)
        for i in range(0, relationsAmount):
            result = 0
            for j in range(0, relationsAmount):
                result += int(cipherText[j]) * int(vector[i * relationsAmount + j])
            relation[0][i] = result % 2
        relationsMatrix = np.vstack((relationsMatrix, relation))
    return relationsMatrix


def verifyResult(args: Namespace, publicKey: List[str], baseVectors: List[np.ndarray], cipherText: List[str]) -> bool:
    """
    Verify the correctness of the solution by calculating the cipher text from the plain text solution and
    matching it to the cipher text from the source file.

    Args:
        publicKey (List[str]): A list of strings representing the public key with placeholders for variables.
        baseVectors (List[numpy.ndarray]): A list of numpy arrays representing the base vectors.
        cipherText (List[str]): A list of strings representing the cipher text.

    Returns:
        bool: True if the calculated cipher text matches the provided cipher text, False otherwise.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Verifying the result')
    
    # Calculate the cipher text with the plain text solution (baseVectors) and the public key
    result = calculateCipherText(args, publicKey, np.array(baseVectors))
    
    # Match the cipher text solution with the cipher text from the *.txt file
    isCorrect = True
    for i in range(0, result.shape[1]):
        if int(result[0][i]) != int(cipherText[i]):
            isCorrect = False
            
    return isCorrect
