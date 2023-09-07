import math
import sys
import numpy
from time import time
from tqdm import tqdm
from os import path
from typing import List, Tuple


def readFile(fileName: str, verbose: bool) -> Tuple[list, list, int]:
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


def generatePlainText(amount: int, verbose: bool) -> numpy.ndarray:
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
                                        numpy.random.choice(a=2, 
                                                            size=(1, amount))))

    return plainTextMatrix


def calculateCipherText(publicKey: List[str], plainTextMatrix: numpy.ndarray, verbose: bool) -> numpy.ndarray:
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


def calculatingMatrix(plainTextMatrix: numpy.ndarray, cipherTextMatrix: numpy.ndarray, verbose: bool) -> numpy.ndarray:
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
        print(f'Calculating matrix from plain and cipher text')

    # Logical AND every single column of a plain text row with every column of the cipher text
    for row in range(0, matrixDimension[0]):
        for plainTextColumn in range(0, plainTextMatrix.shape[1]):
            for cipherTextColumn in range(0, cipherTextMatrix.shape[1]):
                matrix[row][plainTextColumn * plainTextMatrix.shape[1] + cipherTextColumn] = \
                    bool(plainTextMatrix[row][plainTextColumn]) and bool(cipherTextMatrix[row][cipherTextColumn])

    return matrix


# form a matrix into a triangle shape
def gaussElimination(matrix: numpy, amount: int, verbose: bool):
    # eliminate 'zero-rows'
    matrix = matrix[~numpy.all(matrix == 0, axis=1)]

    matrixDimension = matrix.shape
    solvedMatrix = numpy.zeros([0, matrixDimension[1]], dtype=numpy.intc)
    optimalRowSum = math.pow(amount, 2)

    if verbose:
        print(f'start:\tgaussian elimination')

    for column in tqdm(range(0, matrixDimension[1])):
        if matrix.shape[0] > 0:
            # bring all rows with a '1' in column n to the top
            matrix = numpy.flipud(matrix[matrix[:, column].argsort()])
            # find the optimal row (least amount of '1's in the row)
            for row in range(0, matrix.shape[0]):
                currentRowSum = numpy.sum(matrix[row][:])
                if matrix[row][column] == 1 and currentRowSum < optimalRowSum and matrix.shape[0] < 1:
                    optimalRowSum = currentRowSum
                    matrix[column][:] = matrix[row][:]
            # multiply the current pivot element with all followings rows, which contain a '1' in column n
            for followingRow in range(1, matrix.shape[0]):
                if matrix[followingRow][column] == 1:
                    matrix[followingRow] = (matrix[followingRow] + matrix[0]) % 2
            # store the current pivot row in a new output matrix
            solvedMatrix = numpy.vstack([solvedMatrix, matrix[0][:]])
            matrix = numpy.delete(matrix, 0, 0)

    if verbose:
        print(f'end:\ttriangle from completed\n')

    return solvedMatrix


# get every free variable of a matrix
def getFreeVariables(solvedMatrix: numpy, verbose: bool):
    freeVariablesArray = []
    matrixDimension = solvedMatrix.shape

    if verbose:
        print(f'start:\tchecking for free variables')

    print(solvedMatrix)

    # check if column and row is '1', otherwise it's a free variable
    for i in range(0, matrixDimension[0]):
        if solvedMatrix[i][i] != 1:
            freeVariablesArray.append(i + 1)
    # check for additional at the bottom of the matrix
    for i in range(matrixDimension[0], matrixDimension[1]):
        freeVariablesArray.append(i + 1)

    if verbose:
        print(f'end:\tfree variables found\n')

    return freeVariablesArray


# reduce the matrix two its lowest form
def reduceMatrix(matrix, freeVariablesArray, verbose: bool):
    # eliminate 'zero-rows'
    matrix = matrix[~numpy.all(matrix == 0, axis=1)]
    matrixDimension = matrix.shape

    if verbose:
        print(f'start:\treducing matrix')

    # iterate through the columns (except the first)
    for column in tqdm(range(1, matrixDimension[1])):
        # if it is not a pivot column and it is not already reduced
        if column + 1 not in freeVariablesArray and numpy.sum(matrix.T[column]) > 1:
            currentPivotRow = 0
            # iterate upwards through the rows
            for row in range(matrixDimension[0] - 1, -1, -1):
                value = matrix[row][column]
                # find the pivot element
                if value == 1 and currentPivotRow == 0:
                    currentPivotRow = row
                # xor every row above with a '1' with the current pivot row
                elif value == 1:
                    matrix[row] = (matrix[row] + matrix[currentPivotRow]) % 2

    matrix = matrix[~numpy.all(matrix == 0, axis=1)]

    if verbose:
        print(f'end:\tmatrix reduced\n')

    return matrix


# get the base vectors of a matrix
def getBaseVector(solvedMatrix: numpy, freeVariablesArray, verbose: bool):
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
def calculateRelationsMatrix(baseVectorsArray, amount, cipherText, verbose: bool):
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


def executePipeline(inputFile: str, verbose: bool) -> None:
    startingTime = time()

    # Read parameters from the cryptoChallenge and generate matrix
    publicKey, cipherText, amount = readFile(inputFile, verbose)
    PlainTextArray = generatePlainText(amount, verbose)
    cipherTextsArray = calculateCipherText(publicKey, PlainTextArray, verbose)
    matrix = calculatingMatrix(PlainTextArray, cipherTextsArray, verbose)

    # Solve initial matrix
    solvedMatrix = gaussElimination(matrix, amount, verbose)
    freeVariablesArray = getFreeVariables(solvedMatrix, verbose)
    reducedMatrix = reduceMatrix(solvedMatrix, freeVariablesArray, verbose)
    baseVectorsArray = getBaseVector(reducedMatrix, freeVariablesArray, verbose)
    relationsMatrix = calculateRelationsMatrix(baseVectorsArray, amount, cipherText, verbose)

    # Solve matrix consisting of the vectors
    solvedRelationsMatrix = gaussElimination(relationsMatrix, amount, verbose)
    freeVariablesArraySolution = getFreeVariables(solvedRelationsMatrix, verbose)
    reducedRelationsMatrix = reduceMatrix(solvedRelationsMatrix, freeVariablesArraySolution, verbose)
    baseVectorsArraySolution = getBaseVector(reducedRelationsMatrix, freeVariablesArraySolution, verbose)

    # Verify result
    resultcipherText = calculateCipherText(publicKey, numpy.array(baseVectorsArraySolution), verbose)
    isCorrect = True
    for i in range(0, resultcipherText.shape[1]):
        if int(resultcipherText[0][i]) != int(cipherText[i]):
            isCorrect = False

    if verbose:
        print(f'generated PlainText:\n{PlainTextArray}\n\n'
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
    fileName = sys.argv[1]
    verbose = bool(sys.argv[2])

    executePipeline(fileName, verbose)
