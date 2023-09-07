from multiprocessing.dummy import Pool as ThreadPool
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import math
import numpy
from time import time
from tqdm import tqdm


publicKey = []
ciphertextsArray = numpy.zeros([int(2 * math.pow(7, 2)), 7], dtype=numpy.intc)
plaintextsArray = numpy.zeros([int(2 * math.pow(7, 2)), 7], dtype=numpy.intc)


# read any file with a correct structure (publicKey -> ciphertext -> amount)
def readFile(inputFileName: str, verbose: bool):
    if verbose:
        print(f'\nstart:\treading file {inputFileName}')

    with open(inputFileName, 'r') as file:
        text = file.read()

    # Leerzeichen und Zeilenumbrüche weg
    text = text.replace(' ', '').replace('\n', '')

    # Eingrenzen des Strings zur finalen Variable
    publicKey = text.split('[')[1].split(']')[0].split(',')
    ciphertext = text.split('[')[2].split(']')[0].split(',')
    amount = int(text.split('Relationen:')[1])

    if verbose:
        print(f'end:\tfile read\n')

    return publicKey, ciphertext, amount


# generate 2 * amount^2 plaintexts
def generatePlaintext(amount: int, verbose: bool):
    plaintextsAmount = int(2 * math.pow(amount, 2))
    plaintextsArray = numpy.zeros([1, amount], dtype=numpy.intc)

    if verbose:
        print(f'start:\tgenerating {plaintextsAmount} plaintexts')

    # create 2 * amount^2 plaintexts
    while plaintextsArray.shape[0] < int(2 * math.pow(amount, 2)):
        # 2 = Obergrenze -> {0,1}      ;      [1, amount] tupel = maße -> [zeilen, spalten]
        plaintextsArray = numpy.vstack((plaintextsArray, numpy.random.choice(2, [1, amount])))

    if verbose:
        print(f'end:\tplaintexts generated\n')

    return plaintextsArray


# calculate ciphertexts from generated plaintexts
def calculateCiphertext(amount, verbose=False):
    arrayDimensions = ciphertextsArray.shape

    if verbose:
        print(f'start:\tcalculating {arrayDimensions[0]} ciphertexts')

    # for each plaintext, read the value from x_n and replace it with the corresponding variable in the public key
    # after the replacement, calculate the row
    for i in tqdm(range(0, arrayDimensions[0])): # wie viel plaintext durchgehen
        for j, publicRow in enumerate(publicKey): # wie viele zeilen publickey
            for variable in reversed(range(0, arrayDimensions[1])): # rückwärts durchgehen
                # austauschen(ziel, was stattdessen)
                publicRow = publicRow.replace(f'x_{variable + 1}', str(plaintextsArray[i][variable]))
            # fertig zeile ausrechnen
            ciphertextsArray[i][j] = round(eval(publicRow) % 2)

    if verbose:
        print(f'end:\tciphertext calculated\n')

    return ciphertextsArray


# create matrix from plain-/ciphertext pairs
def createMatrix(plaintextsArray: numpy, ciphertextsArray: numpy, verbose: bool):
    # shape gibt tupel zurück [zeile, spalte] -> [0] bedeutet 0. element des tupels
    matrixDimension = plaintextsArray.shape[0], plaintextsArray.shape[1] * ciphertextsArray.shape[1]
    matrix = numpy.zeros(matrixDimension, dtype=numpy.intc)

    if verbose:
        print(f'start:\tcreating matrix')

    # for each row multiply every plaintext column with every ciphertext column
    for row in tqdm(range(0, matrixDimension[0])): # zeilen durchgehen
        for plaintextColumn in range(0, plaintextsArray.shape[1]): # spalten des klartextes durchgehen
            for ciphertextColumn in range(0, ciphertextsArray.shape[1]): # spalten des klartextes durchgehen
                # [0, 1, 2]         -> [3, 4, 5]          -> [6, 7, 8]
                # 0 * 3 + [0, 1, 2]    1 * 3 + [0, 1, 2]     2 * 3 + [0, 1, 2]
                matrix[row][plaintextColumn * plaintextsArray.shape[1] + ciphertextColumn] = \
                    plaintextsArray[row][plaintextColumn] * ciphertextsArray[row][ciphertextColumn]
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


# calculate the relation matrix of the given ciphertext
def calculateRelationsMatrix(baseVectorsArray, amount, ciphertext, verbose: bool):
    relationsMatrix = numpy.zeros([0, amount], dtype=numpy.intc)

    if verbose:
        print(f'start:\tcalculating relation matrix')

    for vector in baseVectorsArray:
        relation = numpy.zeros([1, amount], dtype=numpy.intc)
        # AND operation with the ciphertext (size n) with n parts of the vector
        for i in range(0, amount):
            result = 0
            for j in range(0, amount):
                result += int(ciphertext[j]) * vector[i * amount + j]
            relation[0][i] = result % 2
        relationsMatrix = numpy.vstack((relationsMatrix, relation))

    if verbose:
        print(f'end:\trelation matrix calculated\n')

    return relationsMatrix


def threadChitext(amount, verbose: bool):
    with ThreadPoolExecutor(max_workers=4) as executor:
        future = executor.submit(calculateCiphertext, amount)
        return_value = future.result()
        print(return_value)


# order of events
def routine(inputFile: str, InputVerboseLevel):
    start = time()

    # read and generate matrix
    test, ciphertext, amount = readFile(inputFile, InputVerboseLevel)
    global publicKey
    publicKey = test
    test2 = generatePlaintext(amount, InputVerboseLevel)
    global ciphertextsArray
    ciphertextsArray = test2
    threadChitext(amount / 2, InputVerboseLevel)
    exit()
    #ciphertextsArray = calculateCiphertext(publicKey, plaintextsArray, InputVerboseLevel)
    matrix = createMatrix(plaintextsArray, ciphertextsArray, InputVerboseLevel)

    # solve initial matrix
    solvedMatrix = gaussElimination(matrix, amount, InputVerboseLevel)
    freeVariablesArray = getFreeVariables(solvedMatrix, InputVerboseLevel)
    reducedMatrix = reduceMatrix(solvedMatrix, freeVariablesArray, InputVerboseLevel)
    baseVectorsArray = getBaseVector(reducedMatrix, freeVariablesArray, InputVerboseLevel)
    relationsMatrix = calculateRelationsMatrix(baseVectorsArray, amount, ciphertext, InputVerboseLevel)

    # solve matrix consisting of the vectors
    solvedRelationsMatrix = gaussElimination(relationsMatrix, amount, InputVerboseLevel)
    freeVariablesArraySolution = getFreeVariables(solvedRelationsMatrix, InputVerboseLevel)
    reducedRelationsMatrix = reduceMatrix(solvedRelationsMatrix, freeVariablesArraySolution, InputVerboseLevel)
    baseVectorsArraySolution = getBaseVector(reducedRelationsMatrix, freeVariablesArraySolution, InputVerboseLevel)

    # verify result
    resultCiphertext = calculateCiphertext(publicKey, numpy.array(baseVectorsArraySolution), InputVerboseLevel)
    isCorrect = True
    for i in range(0, resultCiphertext.shape[1]):
        if int(resultCiphertext[0][i]) != int(ciphertext[i]):
            isCorrect = False

    print(f'generated plaintext:\n{plaintextsArray}\n\n'
          f'calculated ciphertext:\n{ciphertextsArray}\n\n'
          f'constructed matrix:\n{matrix}\n\n'
          f'solved matrix:\n{solvedMatrix}\n\n'
          f'reduced matrix:\n{reducedMatrix}\n\n'
          f'free variables:\n{freeVariablesArray}\n\n'
          f'base vectors:\n{baseVectorsArray}\n\n'
          f'relation matrix:\n{relationsMatrix}\n\n'
          f'free variables of the solution:\n{freeVariablesArraySolution}\n\n'
          f'reduced relation matrix of the solution:\n{reducedRelationsMatrix}\n\n'
          f'plaintext solution:\n{baseVectorsArraySolution}\n\n'
          f'chitext of the solution:\n{resultCiphertext}\n\n'
          f'matching:\n{isCorrect}\n\n'
          f'time:\n{time() - start}\n')


if __name__ == '__main__':
    fileName = 'kryptochallengegruppeTest.txt'
    verboseLevel = True

    routine(fileName, verboseLevel)
