import numpy as np
import functions as f
from time import time
from datetime import datetime
from typing import List, Tuple
from argparse import Namespace


def calculateInitialMatrix(args: Namespace) -> Tuple[List[str], List[str], int, np.ndarray]:
    publicKey, cipherText, relationsAmount = f.readFile(args)
    plainTextArray = f.generatePlainText(args, relationsAmount)
    cipherTextsArray = f.calculateCipherText(args, publicKey, plainTextArray)
    matrix = f.calculateMatrix(args, plainTextArray, cipherTextsArray)
    
    return publicKey, cipherText, relationsAmount, matrix


def solveInitialMatrix(args: Namespace, matrix: np.ndarray, relationsAmount: int, cipherText: List[str]) -> np.ndarray:
    solvedMatrix = f.gaussianElimination(args, matrix)
    freeVariables = f.getFreeVariables(args, solvedMatrix)
    reducedMatrix = f.reduceMatrix(args, solvedMatrix, freeVariables)
    baseVectors = f.getBaseVectors(args, reducedMatrix, freeVariables)
    relationsMatrix = f.calculateRelationsMatrix(args, baseVectors, relationsAmount, cipherText)
    
    return relationsMatrix


def solveRelationsMatrix(args: Namespace, relationsMatrix: np.ndarray) -> List[np.ndarray]:
    solvedMatrix = f.gaussianElimination(args, relationsMatrix)
    freeVariables = f.getFreeVariables(args, solvedMatrix)
    reducedMatrix = f.reduceMatrix(args, solvedMatrix, freeVariables)
    baseVectors = f.getBaseVectors(args, reducedMatrix, freeVariables)
    
    return baseVectors


def executePipeline(args: Namespace) -> None:
    """
    Execute a pipeline to solve the cryptographic problem using various matrix operations.
    """
    startingTime = time()

    # Execute the pipeline
    publicKey, cipherText, relationsAmount, matrix = calculateInitialMatrix(args)
    relationsMatrix = solveInitialMatrix(args, matrix, relationsAmount, cipherText)
    baseVectors = solveRelationsMatrix(args, relationsMatrix)

    # Verify the result to insure that the calculation was right
    isCorrect = f.verifyResult(args, publicKey, baseVectors, cipherText)
    
    currentTime = datetime.now().strftime("%H:%M:%S")  
    if isCorrect:
        print(f'[{currentTime}] Successful! Plain text solution: {np.array(baseVectors, dtype=np.uint8)}')
        if args.verbose:
              print(f'[{currentTime}] Solved in: {round((time() - startingTime), 2)} seconds')
    else:
        print(f'[{currentTime}] Decryption failed')
