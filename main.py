import numpy
from sys import argv
from time import time
from math import pow as mathPow
from typing import List, Tuple
from datetime import datetime


def readFile(filePath: str) -> Tuple[List[str], List[str], int]:
    """
    Read and process an input file into 3 variables.

    Args:
        filePath (str): The path of the file to be read.

    Returns:
        Tuple[list, list, int]: A tuple containing the public key (list), 
                                cipher text (list), and relationsAmount (int) extracted 
                                from the file.
    """
    if verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Processing file \'{filePath}\'')

    # Read the given file.
    with open(filePath, 'r') as file:
        text = file.read()

    # Remove blank spaces and linebreaks for easier processing.
    text = text.replace(' ', '').replace('\n', '')

    # Separating the text into its variables.
    try:
        publicKey = text.split('[')[1].split(']')[0].split(',')
        cipherText = text.split('[')[2].split(']')[0].split(',')
        relationsAmount = int(text.split('relations:')[1])
    except:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Unable to locate all parameters from the file {filePath}.\n'
              f'  Check for the right formatting.')
        exit(-1)

    return publicKey, cipherText, relationsAmount


def generatePlainText(relationsAmount: int) -> numpy.ndarray:
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

    if verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Generating {round(plainTextAmount)} plain texts')

    plainTextMatrix = numpy.zeros(shape=(1, relationsAmount), 
                                  dtype=numpy.bool_)

    # Generate random plain texts until 2 * relationsAmount² rows were generated.
    while plainTextMatrix.shape[0] < plainTextAmount:
        plainTextMatrix = numpy.vstack((plainTextMatrix, 
                                        numpy.random.choice(a=numpy.array([True, False]), 
                                                            size=(1, relationsAmount))))

    return plainTextMatrix


def calculateCipherText(publicKey: List[str], plainTextMatrix: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate cipher texts using the public key and plain text matrix.

    Args:
        publicKey (List[str]): List of strings representing the public key with variable placeholders.
        plainTextMatrix (numpy.ndarray): 2D numpy array containing plain text values.

    Returns:
        numpy.ndarray: 2D numpy array containing calculated cipher text values.
    """
    arrayDimensions = plainTextMatrix.shape
    cipherTextMatrix = numpy.zeros(shape=arrayDimensions, 
                                   dtype=numpy.bool_)

    if verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Calculating {arrayDimensions[0]} corresponding cipher texts')

    # Replace the variables x_n of the public key with the corresponding plain text values to calculate the cipher text
    for row in range(0, arrayDimensions[0]):
        for column, publicKeyRow in enumerate(publicKey):
            for variable in reversed(range(0, arrayDimensions[1])):
                publicKeyRow = publicKeyRow.replace(f'x_{variable + 1}', str(plainTextMatrix[row][variable]))
            cipherTextMatrix[row][column] = eval(publicKeyRow) % 2

    return cipherTextMatrix


def calculatingMatrix(plainTextMatrix: numpy.ndarray, cipherTextMatrix: numpy.ndarray) -> numpy.ndarray:
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
    matrix = numpy.zeros(shape=matrixDimension, 
                         dtype=numpy.bool_)

    if verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Calculating matrix from plain and cipher text')

    # Logical AND every single column of a plain text row with every column of the cipher text
    for row in range(0, matrixDimension[0]):
        for plainTextColumn in range(0, plainTextMatrix.shape[1]):
            for cipherTextColumn in range(0, cipherTextMatrix.shape[1]):
                matrix[row][plainTextColumn * plainTextMatrix.shape[1] + cipherTextColumn] = \
                    bool(plainTextMatrix[row][plainTextColumn]) and bool(cipherTextMatrix[row][cipherTextColumn])

    return matrix


def gaussianElimination(matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Perform Gaussian elimination on a binary matrix to simplify and solve the system of equations.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing a binary matrix.

    Returns:
        numpy.ndarray: A 2D numpy array representing the simplified matrix after Gaussian elimination.
    """
    if verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Starting gaussian elimination')

    # Remove every duplicate and False/zero only rows
    matrix = numpy.unique(ar=matrix, 
                          axis=0)
    matrix = matrix[~numpy.all(matrix == False, 
                               axis=1)]
    
    solvedMatrix = numpy.zeros(shape=[0, matrix.shape[1]], 
                               dtype=numpy.bool_)

    for column in range(0, matrix.shape[1]):
        if matrix.shape[0] > 1:
            # Move all rows with a True/one in the nth column to the top of the matrix
            matrix = numpy.flipud(matrix[matrix[:, column].argsort()])
            
            # Logical XOR the current pivot row with all followings rows, which contain a True/one in the nth column
            if matrix[0][column] == True:
                for row in range(1, matrix.shape[0]):
                    if matrix[row][column] == True:
                        matrix[row][:] = numpy.logical_xor(matrix[0][:], matrix[row][:])
                    else:
                        break
            else:
                continue
        
        if matrix.shape[0] > 0:  
            # Store the current pivot row in the output matrix and delete the same row in the input matrix
            solvedMatrix = numpy.vstack([solvedMatrix, matrix[0][:]])
            matrix = numpy.delete(arr=matrix, 
                                  obj=0, 
                                  axis=0)
        else:
            break
        
    solvedMatrix = solvedMatrix[~numpy.all(solvedMatrix == False, 
                                           axis=1)]

    return solvedMatrix


def getFreeVariables(matrix: numpy.ndarray) -> List[int]:
    """
    Find and return the indices of free variables in the solved binary matrix.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing the simplified matrix after Gaussian elimination.

    Returns:
        List[int]: A list of integers representing the indices of free variables.
    """
    if verbose:
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


def reduceMatrix(matrix: numpy.ndarray, freeVariables: List[int]) -> numpy.ndarray:
    """
    Reduce a binary matrix by performing additional operations based on free variables.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing a binary matrix.
        freeVariables (List[int]): A list of integers representing the indices of free variables.

    Returns:
        numpy.ndarray: A 2D numpy array representing the reduced binary matrix.
    """
    if verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Reducing matrix')

    for column in range(1, matrix.shape[1]):
        # If the column is not a free variable and not fully reduced
        if column not in freeVariables and numpy.sum(matrix.T[column]) > 1:
            currentPivotRow = 0
            # Iterate upwards through the rows
            for row in range(matrix.shape[0] - 1, -1, -1):
                value = matrix[row][column]
                # Find the pivot element and logical XOR every row above containing a True/one
                if value == True:
                    if currentPivotRow == 0:
                        currentPivotRow = row
                    else:
                        matrix[row][:] = numpy.logical_xor(matrix[row][:], matrix[currentPivotRow][:])

    matrix = matrix[~numpy.all(matrix == False, 
                               axis=1)]

    return matrix


def getBaseVectors(matrix: numpy.ndarray, freeVariables: List[int]) -> List[numpy.ndarray]:
    """
    Find and return the base vectors from a binary matrix based on free variables.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing a binary matrix.
        freeVariables (List[int]): A list of integers representing the indices of free variables.

    Returns:
        List[numpy.ndarray]: A list of numpy arrays representing the base vectors.
    """
    if verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Searching for base vectors')
    
    # Insert extra rows to extract complete vectors
    for variable in freeVariables:
        matrix = numpy.insert(arr=matrix, 
                              obj=variable,
                              values=numpy.zeros(shape=[1, matrix.shape[1]], 
                                                 dtype=numpy.bool_), 
                              axis=0)
        matrix[variable][variable] = 1

    # Save all columns which represent free variables
    baseVectors = []
    matrix = matrix.T
    for variable in reversed(freeVariables):
        baseVectors.append(matrix[:][variable])

    return baseVectors


def calculateRelationsMatrix(baseVectors: List[numpy.ndarray], relationsAmount: int, cipherTextMatrix: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate a relations matrix based on base vectors and a cipher text matrix.

    Args:
        baseVectors (List[numpy.ndarray]): A list of numpy arrays representing base vectors.
        relationsAmount (int): The number of relations to calculate.
        cipherTextMatrix (numpy.ndarray): A 2D numpy array representing a cipher text matrix.

    Returns:
        numpy.ndarray: A 2D numpy array representing the relations matrix.
    """
    if verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Calculating relations matrix')
    
    relationsMatrix = numpy.zeros(shape=[0, relationsAmount], 
                                  dtype=numpy.bool_)

    # Logical AND every cipherText (of size 'relationsAmount') with 'relationsAmount' values of the vector
    for vector in baseVectors:
        relation = numpy.zeros(shape=[1, relationsAmount], 
                               dtype=numpy.bool_)
        for i in range(0, relationsAmount):
            result = 0
            for j in range(0, relationsAmount):
                result += int(cipherTextMatrix[j][:]) * int(vector[i * relationsAmount + j])
            relation[0][i] = result % 2
        relationsMatrix = numpy.vstack((relationsMatrix, relation))

    return relationsMatrix


def verifyResult(publicKey: List[str], baseVectors: List[numpy.ndarray], cipherText: List[str]) -> bool:
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
    if verbose:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Verifying the result')
    
    # Calculate the cipher text with the plain text solution (baseVectors)
    result = calculateCipherText(publicKey, numpy.array(baseVectors))
    
    # Match the cipher text solution with the cipher text from the *.txt file
    isCorrect = True
    for i in range(0, result.shape[1]):
        if int(result[0][i]) != int(cipherText[i]):
            isCorrect = False
            
    return isCorrect


def executePipeline(filePath: str) -> None:
    """
    Execute a pipeline to solve the cryptographic problem using various matrix operations.

    Args:
        inputFile (str): The path of the input file containing the parameters.
    """
    startingTime = time()

    # Read the parameters from a formatted *.txt file and calculate a matrix to solve
    publicKey, cipherText, relationsAmount = readFile(filePath)
    plainTextArray = generatePlainText(relationsAmount)
    cipherTextsArray = calculateCipherText(publicKey, plainTextArray)
    matrix = calculatingMatrix(plainTextArray, cipherTextsArray)

    # Solve the initial matrix
    solvedMatrix = gaussianElimination(matrix)
    freeVariables = getFreeVariables(solvedMatrix)
    reducedMatrix = reduceMatrix(solvedMatrix, freeVariables)
    baseVectors = getBaseVectors(reducedMatrix, freeVariables)
    relationsMatrix = calculateRelationsMatrix(baseVectors, relationsAmount, cipherText)

    # Solve the relations matrix
    solvedRelationsMatrix = gaussianElimination(relationsMatrix)
    freeVariables = getFreeVariables(solvedRelationsMatrix)
    reducedRelationsMatrix = reduceMatrix(solvedRelationsMatrix, freeVariables)
    baseVectors = getBaseVectors(reducedRelationsMatrix, freeVariables)

    # Verify the result to insure that the calculation was right
    isCorrect = verifyResult(publicKey, baseVectors, cipherText)
    
    currentTime = datetime.now().strftime("%H:%M:%S")  
    if isCorrect:
        print(f'[{currentTime}] Plain text solution: {numpy.array(baseVectors, dtype=numpy.uint8)}\n'
              f'[{currentTime}] Solved in: {round((time() - startingTime), 2)} seconds')
    else:
        print(f'[{currentTime}] Decryption failed')


if __name__ == '__main__':
    if len(argv) < 3:
        print('Usage: python main.py \'filePath\' verbose\n'
              '       - filePath: Full path of the text file\n'
              '       - verbose:  0 or 1 to print extra information')
        exit(-1)
    
    verbose = bool(argv[2])
    filePath = argv[1]

    executePipeline(filePath)
