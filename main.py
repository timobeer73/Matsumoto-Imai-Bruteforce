import math
import sys
import numpy
from time import time
from tqdm import tqdm
from os import path
from typing import Tuple


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
        print(f'Generating {plainTextAmount} plain texts')

    plainTextArray = numpy.zeros(shape=(1, amount), 
                                 dtype=numpy.bool_)

    # Generate random plain texts until 2 * amount² rows were generated.
    while plainTextArray.shape[0] < plainTextAmount:
        plainTextArray = numpy.vstack((plainTextArray, 
                                       numpy.random.choice(a=2, 
                                                           size=(1, amount))))

    return plainTextArray


# calculate cipherTexts from generated PlainText
def calculateCipherText(publicKey: list[str], PlainTextArray: numpy, verbose: bool):
    arrayDimensions = PlainTextArray.shape
    cipherTextArray = numpy.zeros(arrayDimensions, dtype=numpy.intc)

    if verbose:
        print(f'start:\tcalculating {arrayDimensions[0]} cipherTexts')

    # for each PlainText, read the value from x_n and replace it with the corresponding variable in the public key
    # after the replacement, calculate the row
    for i in tqdm(range(0, arrayDimensions[0])): # wie viel PlainText durchgehen
        for j, publicRow in enumerate(publicKey): # wie viele zeilen publickey
            for variable in reversed(range(0, arrayDimensions[1])): # rückwärts durchgehen
                # austauschen(ziel, was stattdessen)
                publicRow = publicRow.replace(f'x_{variable + 1}', str(PlainTextArray[i][variable]))
            # fertig zeile ausrechnen
            cipherTextArray[i][j] = round(eval(publicRow) % 2)

    if verbose:
        print(f'end:\tcipherText calculated\n')

    return cipherTextArray


# create matrix from plain-/cipherText pairs
def createMatrix(PlainTextArray: numpy, cipherTextsArray: numpy, verbose: bool):
    # shape gibt tupel zurück [zeile, spalte] -> [0] bedeutet 0. element des tupels
    matrixDimension = PlainTextArray.shape[0], PlainTextArray.shape[1] * cipherTextsArray.shape[1]
    matrix = numpy.zeros(matrixDimension, dtype=numpy.intc)

    if verbose:
        print(f'start:\tcreating matrix')

    # for each row multiply every PlainText column with every cipherText column
    for row in tqdm(range(0, matrixDimension[0])): # zeilen durchgehen
        for PlainTextColumn in range(0, PlainTextArray.shape[1]): # spalten des klartextes durchgehen
            for cipherTextColumn in range(0, cipherTextsArray.shape[1]): # spalten des klartextes durchgehen
                # [0, 1, 2]         -> [3, 4, 5]          -> [6, 7, 8]
                # 0 * 3 + [0, 1, 2]    1 * 3 + [0, 1, 2]     2 * 3 + [0, 1, 2]
                matrix[row][PlainTextColumn * PlainTextArray.shape[1] + cipherTextColumn] = \
                    PlainTextArray[row][PlainTextColumn] * cipherTextsArray[row][cipherTextColumn]
    # [1, 2, 3]
    # [4, 5, 8]
    # [1*4, 1*5, ...]

    if verbose:
        print(f'end:\tcreated matrix\n')

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
    matrix = createMatrix(PlainTextArray, cipherTextsArray, verbose)

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
