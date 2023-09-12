import math
import sys
import numpy
from time import time
from tqdm import tqdm
from os import path
from typing import List, Tuple


def readFile(fileName: str) -> Tuple[list, list, int]:
    """
    Read and process an input file into 3 variables.

    Args:
        fileName (str): The name of the file to be read.
        verbose (bool): Whether to print verbose messages.

    Returns:
        Tuple[list, list, int]: A tuple containing the public key (list), 
                                cipher text (list), and amount (int) extracted 
                                from the file.
    """
    if verbose:
        print(f'Processing file \'{fileName}\'')

    # Locate and read the given file.
    folderPath = path.dirname(__file__)
    filePath = path.join(folderPath, fileName)
    with open(filePath, 'r') as file:
        text = file.read()

    # Remove blank spaces and linebreaks for easier processing.
    text = text.replace(' ', '').replace('\n', '')

    # Separating the text into its variables.
    publicKey = text.split('[')[1].split(']')[0].split(',')
    cipherText = text.split('[')[2].split(']')[0].split(',')
    amount = int(text.split('relations:')[1])

    return publicKey, cipherText, amount


def generatePlainText(amount: int) -> numpy.ndarray:
    """
    Generate a 2D numpy array of random plain texts.

    Args:
        amount (int): The number of basic elements/special relations.
        verbose (bool): Whether to print verbose messages.

    Returns:
        numpy.ndarray: A 2D numpy array of shape (2 * amount², amount) 
                       containing random plain texts represented by 
                       binary values.
    """
    plainTextAmount = 2 * math.pow(amount, 2)

    if verbose:
        print(f'Generating {round(plainTextAmount)} plain texts')

    plainTextMatrix = numpy.zeros(shape=(1, amount), 
                                  dtype=numpy.bool_)

    # Generate random plain texts until 2 * amount² rows were generated.
    while plainTextMatrix.shape[0] < plainTextAmount:
        plainTextMatrix = numpy.vstack((plainTextMatrix, 
                                        numpy.random.choice(a=numpy.array([True, False]), 
                                                            size=(1, amount))))

    return plainTextMatrix


def calculateCipherText(publicKey: List[str], plainTextMatrix: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate the corresponding cipher texts using the public key and plain text array.

    Args:
        publicKey (List[str]): A list of strings representing the public key with placeholders for variables.
        plainTextMatrix (numpy.ndarray): A 2D numpy array containing the plain text values.
        verbose (bool): Whether to print verbose messages.

    Returns:
        numpy.ndarray: A 2D numpy array containing the calculated cipher text values.
    """
    arrayDimensions = plainTextMatrix.shape
    cipherTextMatrix = numpy.zeros(shape=arrayDimensions, 
                                   dtype=numpy.bool_)

    if verbose:
        print(f'Calculating {arrayDimensions[0]} corresponding cipher texts')

    # Replace the variables x_n of the public key with the corresponding plain text values to 
    # calculate the cipher text
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
        verbose (bool): Whether to print verbose messages.

    Returns:
        numpy.ndarray: A 2D numpy array containing the result of logical AND operations between
                       corresponding elements of the input matrices.
    """
    matrixDimension = plainTextMatrix.shape[0], plainTextMatrix.shape[1] * cipherTextMatrix.shape[1]
    matrix = numpy.zeros(shape=matrixDimension, 
                         dtype=numpy.bool_)

    if verbose:
        print('Calculating matrix from plain and cipher text')

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
        print('Starting gaussian elimination')

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
        print('Searching for free variables')

    # Check if the ith column and row is True/one. If not add it to the free variables
    freeVariables = []
    for i in range(0, matrix.shape[0]):
        if matrix[i][i] == False:
            freeVariables.append(i)
    
    # Add additional free variables beyond the current matrix row size
    for i in range(matrix.shape[0], matrix.shape[1]):
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
        print('Reducing matrix')

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


# get the base vectors of a matrix
def getBaseVector(solvedMatrix: numpy, freeVariablesArray):
    baseVectorsArray = []
    temporaryMatrix = solvedMatrix
    temporaryMatrix = temporaryMatrix[~numpy.all(temporaryMatrix == 0, axis=1)]

    if verbose:
        print(f'start:\tgetting the base vectors')

    # insert extra rows for easier vector extraction
    for variable in freeVariablesArray:
        temporaryMatrix = numpy.insert(temporaryMatrix, variable - 1,
                                       numpy.zeros([1, temporaryMatrix.shape[1]], dtype=numpy.intc), 0)
        temporaryMatrix[variable - 1][variable - 1] = 1

    # save all columns of a free variable
    temporaryMatrix = temporaryMatrix.T
    for variable in reversed(freeVariablesArray):
        baseVectorsArray.append(temporaryMatrix[:][variable - 1])

    if verbose:
        print(f'end:\tbase vectors found\n')

    return baseVectorsArray


# calculate the relation matrix of the given cipherText
def calculateRelationsMatrix(baseVectorsArray, amount, cipherText):
    relationsMatrix = numpy.zeros([0, amount], dtype=numpy.intc)

    if verbose:
        print(f'start:\tcalculating relation matrix')

    for vector in baseVectorsArray:
        relation = numpy.zeros([1, amount], dtype=numpy.intc)
        # AND operation with the cipherText (size n) with n parts of the vector
        for i in range(0, amount):
            result = 0
            for j in range(0, amount):
                result += int(cipherText[j]) * vector[i * amount + j]
            relation[0][i] = result % 2
        relationsMatrix = numpy.vstack((relationsMatrix, relation))

    if verbose:
        print(f'end:\trelation matrix calculated\n')

    return relationsMatrix


def executePipeline(inputFile: str) -> None:
    startingTime = time()

    # Read parameters from the cryptoChallenge and generate matrix
    publicKey, cipherText, amount = readFile(inputFile)
    numpy.set_printoptions(threshold=sys.maxsize)
    plainTextArray = generatePlainText(amount)
    cipherTextsArray = calculateCipherText(publicKey, plainTextArray)
    matrix = calculatingMatrix(plainTextArray, cipherTextsArray)

    # Solve initial matrix
    solvedMatrix = gaussianElimination(matrix)
    print(solvedMatrix)
    exit()
    freeVariablesArray = getFreeVariables(solvedMatrix)
    reducedMatrix = reduceMatrix(solvedMatrix, freeVariablesArray)
    baseVectorsArray = getBaseVector(reducedMatrix, freeVariablesArray)
    relationsMatrix = calculateRelationsMatrix(baseVectorsArray, amount, cipherText)

    # Solve matrix consisting of the vectors
    solvedRelationsMatrix = gaussianElimination(relationsMatrix, amount)
    freeVariablesArraySolution = getFreeVariables(solvedRelationsMatrix)
    reducedRelationsMatrix = reduceMatrix(solvedRelationsMatrix, freeVariablesArraySolution)
    baseVectorsArraySolution = getBaseVector(reducedRelationsMatrix, freeVariablesArraySolution)

    # Verify result
    resultcipherText = calculateCipherText(publicKey, numpy.array(baseVectorsArraySolution))
    isCorrect = True
    for i in range(0, resultcipherText.shape[1]):
        if int(resultcipherText[0][i]) != int(cipherText[i]):
            isCorrect = False

    if verbose:
        print(f'generated PlainText:\n{plainTextArray}\n\n'
              f'calculated cipherText:\n{cipherTextsArray}\n\n'
              f'constructed matrix:\n{matrix}\n\n'
              f'solved matrix:\n{solvedMatrix}\n\n'
              f'reduced matrix:\n{reducedMatrix}\n\n'
              f'free variables:\n{freeVariablesArray}\n\n'
              f'base vectors:\n{baseVectorsArray}\n\n'
              f'relation matrix:\n{relationsMatrix}\n\n'
              f'free variables of the solution:\n{freeVariablesArraySolution}\n\n'
              f'reduced relation matrix of the solution:\n{reducedRelationsMatrix}\n\n'
              f'PlainText solution:\n{baseVectorsArraySolution}\n\n'
              f'chitext of the solution:\n{resultcipherText}\n\n'
              f'matching:\n{isCorrect}\n\n'
              f'time:\n{time() - startingTime}\n')
    else:
        print(baseVectorsArraySolution)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python main.py \'fileName\' verbose')
        exit(-1)
    
    verbose = bool(sys.argv[2])
    fileName = sys.argv[1]

    executePipeline(fileName)
