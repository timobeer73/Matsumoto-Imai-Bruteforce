import numpy as np
from datetime import datetime
from typing import List, Tuple
from argparse import Namespace
from math import pow as mathPow
from multiprocessing import Pool, cpu_count


def readFile(args: Namespace) -> Tuple[List[str], List[str], int]:
    """
    Read and process an input file into three variables: the public key used for the 
    encryption, an intercepted cipher text and the amount of relations.

    Returns:
        Tuple[List[str], List[str], int]: A tuple containing the public key, cipher text
                                          and relationsAmount extracted from the file.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Processing file \'{args.filepath}\'')

    # Read the given file and save the content into a string.
    with open(args.filepath, 'r') as file:
        fileContent = file.read()

    # Remove blank spaces and linebreaks for easier processing.
    fileContent = fileContent.replace(' ', '').replace('\n', '')

    try:
        # Separating the text into its variables.
        publicKey = fileContent.split('[')[1].split(']')[0].split(',')
        cipherText = fileContent.split('[')[2].split(']')[0].split(',')
        # Derive the amount of relations from the length of the cipher text.
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
        np.ndarray: A 2D numpy array of shape (2 * relationsAmount, relationsAmount)
                    containing random plain texts represented by binary values.
    """
    plainTextAmount = int(2 * mathPow(relationsAmount, 2))

    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Generating {round(plainTextAmount)} plain texts')

    plainTextMatrix = np.zeros(shape=(1, relationsAmount), 
                               dtype=np.bool_)

    # Generate 2 * relationsAmount² rows of random plain text, sufficient for constructing a relation matrix.
    for _ in range(plainTextAmount):
        plainTextMatrix = np.vstack((plainTextMatrix, 
                                     np.random.choice(a=np.array([True, False]), 
                                                      size=(1, relationsAmount))))

    return plainTextMatrix


def calculateCipherTextRow(args, publicKey, rowSize, cipherTextMatrixRow, plainTextMatrixRow):
    # Replace the placeholder variables (x_n) of the public key with the corresponding plain text values.
    for columnIndex, publicKeyRow in enumerate(publicKey):
        for variableIndex in reversed(range(0, rowSize)):
            publicKeyRow = publicKeyRow.replace(f'x_{variableIndex + 1}', str(plainTextMatrixRow[variableIndex]))
        # Calculate the constructed formula.
        cipherTextMatrixRow[columnIndex] = eval(publicKeyRow) % 2
    
    return cipherTextMatrixRow


def calculateCipherTextRowWrapper(args):
    return calculateCipherTextRow(*args)


def calculateCipherText(args: Namespace, publicKey: List[str], plainTextMatrix: np.ndarray) -> np.ndarray:
    """
    Calculate the corresponding cipher text using the public key and plain text matrix.

    Args:
        publicKey (List[str]): A list of strings representing the public key with variable placeholders.
        plainTextMatrix (np.ndarray): A 2D numpy array containing plain text.

    Returns:
        np.ndarray: A 2D numpy array containing the corresponding cipher text.
    """
    matrixDimensions = plainTextMatrix.shape
    cipherTextMatrix = np.zeros(shape=matrixDimensions, 
                                dtype=np.bool_)

    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Calculating {matrixDimensions[0]} corresponding cipher texts')

    with Pool(processes=min(matrixDimensions[0], cpu_count())) as pool:
        result = pool.map(calculateCipherTextRowWrapper, [(args, publicKey, matrixDimensions[1], cipherTextMatrix[rowIndex][:], plainTextMatrix[rowIndex][:]) for rowIndex in range(matrixDimensions[0])])
    
    cipherTextMatrix = np.array(result)

    return cipherTextMatrix


def calculateMatrix(args: Namespace, plainTextMatrix: np.ndarray, cipherTextMatrix: np.ndarray) -> np.ndarray:
    """
    Calculate a matrix by performing logical AND operations between the plain text and cipher text matrices.

    Args:
        plainTextMatrix (np.ndarray): A 2D numpy array containing plain text.
        cipherTextMatrix (np.ndarray): A 2D numpy array containing cipher text.

    Returns:
        np.ndarray: A 2D numpy array containing the result of the logical AND operations between
                    corresponding elements of the input matrices.
    """
    matrixDimension = plainTextMatrix.shape[0], plainTextMatrix.shape[1] * cipherTextMatrix.shape[1]
    matrix = np.zeros(shape=matrixDimension, 
                      dtype=np.bool_)

    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Calculating matrix from plain and cipher text')

    # Logical AND every single column of the plain text row with every column of the cipher text.
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
        matrix (np.ndarray): A 2D numpy array representing a binary matrix.

    Returns:
        np.ndarray: A 2D numpy array representing the simplified matrix after Gaussian elimination.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Starting gaussian elimination')
        
    # Remove all duplicates and every row which only contains False/zeros.
    matrix = np.unique(ar=matrix, 
                       axis=0)
    matrix = matrix[~np.all(matrix == False, 
                            axis=1)]
    
    solvedMatrix = np.zeros(shape=[0, matrix.shape[1]], 
                            dtype=np.bool_)
    for columnIndex in range(0, matrix.shape[1]):
        # Move all rows with a True/one in the nth column to the top of the matrix.
        matrix = np.flipud(matrix[matrix[:, columnIndex].argsort()])
        if matrix[0][columnIndex] == True:
            if matrix.shape[0] > 1:
                # Move the row with the lowest amount of True/ones to the top.
                optimalRowSum = matrix.shape[1]
                optimalRowIndex = 0
                for rowIndex, row in enumerate(matrix):
                    if row[columnIndex] == True:
                        rowSum = np.count_nonzero(row)
                        if rowSum < optimalRowSum:
                            optimalRowSum = rowSum
                            optimalRowIndex = rowIndex
                    else:
                        break
                if optimalRowIndex != 0:
                    matrix[[0, optimalRowIndex]] = matrix[[optimalRowIndex, 0]]

                # Logical XOR the current pivot row with all followings rows, which contain a 
                # True/one in the nth column.
                for rowIndex in range(1, matrix.shape[0]):
                    if matrix[rowIndex][columnIndex] == True:
                        matrix[rowIndex][:] = np.logical_xor(matrix[0][:], matrix[rowIndex][:])
                    else:
                        break
            
            if matrix.shape[0] > 0:  
                # Store the current pivot row in the output matrix and delete the same row in the input matrix.
                solvedMatrix = np.vstack([solvedMatrix, matrix[0][:]])
                matrix = np.delete(arr=matrix, 
                                obj=0, 
                                axis=0)
            else:
                break
            
    matrix = np.unique(ar=matrix, 
                       axis=0)
    matrix = matrix[~np.all(matrix == False, 
                            axis=1)]
    
    return solvedMatrix


def getFreeVariables(args: Namespace, matrix: np.ndarray) -> List[int]:
    """
    Find and return the indices of free variables in the solved binary matrix.

    Args:
        matrix (np.ndarray): A 2D numpy array representing the simplified matrix after Gaussian elimination.

    Returns:
        List[int]: A list of integers representing the indices of free variables.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Searching for free variables')

    freeVariables = []
    offset = 0
    for columnIndex in range(0, matrix.shape[1]):
        if (columnIndex - offset) == matrix.shape[0]:
            break
        # Check if the value is False and add it to the free variables.
        elif matrix[columnIndex - offset][columnIndex] == False:
            freeVariables.append(columnIndex)
            offset += 1
    
    # Add additional free variables beyond the current matrix row size.
    for columnIndex in range(matrix.shape[0] + offset, matrix.shape[1]):
        freeVariables.append(columnIndex)

    return freeVariables


def reduceMatrix(args: Namespace, matrix: np.ndarray, freeVariables: List[int]) -> np.ndarray:
    """
    Reduce a binary matrix by performing logical XOR operations between the pivot row and
    overlying rows containing a True/one in the same column as the current pivot element.

    Args:
        matrix (np.ndarray): A 2D numpy array representing a binary matrix.
        freeVariables (List[int]): A list of integers representing the indices of free variables.

    Returns:
        np.ndarray: A 2D numpy array representing the reduced binary matrix.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Reducing matrix')

    offset = 0
    for column in range(1, matrix.shape[1]):
        # If the column is not a free variable and not fully reduced.
        if column not in freeVariables and np.sum(matrix.T[column]) > 1:
            # Iterate upwards and search for a True/one in the specific column.
            for overlyingRowIndex in range(column - offset - 1, -1, -1):
                if matrix[overlyingRowIndex][column] == True:
                    # Logical XOR the rows matching the previous criteria.
                    matrix[overlyingRowIndex][:] = np.logical_xor(matrix[overlyingRowIndex][:], matrix[column - offset][:])
        if column in freeVariables:
            offset += 1

    matrix = matrix[~np.all(matrix == False, 
                            axis=1)]
    
    return matrix


def getBaseVectors(args: Namespace, matrix: np.ndarray, freeVariables: List[int]) -> List[np.ndarray]:
    """
    Find and return the base vectors from a binary matrix based on free variables.

    Args:
        matrix (np.ndarray): A 2D numpy array representing a binary matrix.
        freeVariables (List[int]): A list of integers representing the indices of free variables.

    Returns:
        List[np.ndarray]: A list of numpy arrays representing the base vectors.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Searching for base vectors')
    
    # Insert extra rows to the reduced matrix to extract the complete vector.
    for variableIndex in freeVariables:
        matrix = np.insert(arr=matrix, 
                           obj=variableIndex,
                           values=np.zeros(shape=[1, matrix.shape[1]], 
                                           dtype=np.bool_), 
                           axis=0)
        matrix[variableIndex][variableIndex] = 1

    # Save the content of all columns with the index specified in the freeVariables variable.
    baseVectors = []
    matrix = matrix.T
    for variableIndex in reversed(freeVariables):
        baseVectors.append(matrix[:][variableIndex])

    return baseVectors


def calculateRelationsMatrix(args: Namespace, baseVectors: List[np.ndarray], relationsAmount: int, cipherText: List[str]) -> np.ndarray:
    """
    Calculate a relations matrix based on base vectors and the cipher text.

    Args:
        baseVectors (List[np.ndarray]): A list of numpy arrays representing base vectors.
        relationsAmount (int): The number of relations.
        cipherText (List[str]): An array representing the cipher text.

    Returns:
        np.ndarray: A 2D numpy array representing the relations matrix.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Calculating relations matrix')
    
    relationsMatrix = np.zeros(shape=[0, relationsAmount], 
                               dtype=np.bool_)

    # Logical AND every position of the ciphertext with 'relationsAmount'-large parts of the base vectors.
    for vector in baseVectors:
        relation = np.zeros(shape=[1, relationsAmount], 
                            dtype=np.bool_)
        for offset in range(0, relationsAmount):
            result = 0
            for column in range(0, relationsAmount):
                result += int(cipherText[column]) * int(vector[offset * relationsAmount + column])
            relation[0][offset] = result % 2
        relationsMatrix = np.vstack((relationsMatrix, relation))
    
    return relationsMatrix


def verifyResult(args: Namespace, publicKey: List[str], baseVectors: List[np.ndarray], cipherText: List[str]) -> bool:
    """
    Verify the correctness of the solution by calculating the cipher text from the plain text solution and
    matching it to the cipher text from the source file.

    Args:
        publicKey (List[str]): A list of strings representing the public key with placeholders for variables.
        baseVectors (List[np.ndarray]): A list of numpy arrays representing the base vectors.
        cipherText (List[str]): A list of strings representing the cipher text.

    Returns:
        bool: True if the calculated cipher text matches the provided cipher text, False otherwise.
    """
    if args.verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Verifying the result')
    
    # Calculate the cipher text with the plain text solution (baseVectors) and the public key.
    result = calculateCipherText(args, publicKey, np.array(baseVectors))
    
    # Match the cipher text solution with the cipher text from the *.txt file.
    isCorrect = True
    for column in range(0, result.shape[1]):
        if int(result[0][column]) != int(cipherText[column]):
            isCorrect = False
            
    return isCorrect
