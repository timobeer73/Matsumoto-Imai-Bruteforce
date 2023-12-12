import numpy as np
import functions as f
from typing import List, Tuple, Union
from argparse import Namespace


def generateMatrix(args: Namespace) -> Tuple[List[str], List[str], int, np.ndarray]:
    """
    Calculate the initial matrices and values required for the cryptographic problem.

    Returns:
        Tuple[List[str], List[str], int, np.ndarray]: A tuple containing the public key,
                                                      cipher text, relationsAmount, and the
                                                      initial matrix.
    """
    publicKey, cipherText, relationsAmount = f.readFile(args)
    plainTextArray = f.generatePlainText(args, relationsAmount)
    cipherTextsArray = f.calculateCipherText(args, publicKey, plainTextArray)
    matrix = f.calculateMatrix(args, plainTextArray, cipherTextsArray)

    return publicKey, cipherText, relationsAmount, matrix


def solveMatrix(args: Namespace, matrix: np.ndarray, relationsAmount: int, cipherText: List[str], createRelationsMatrix: bool) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Solve the given matrix and obtain the relations matrix or base vectors.

    Args:
        matrix (np.ndarray): The initial matrix.
        relationsAmount (int): The number of basic elements/special relations.
        cipherText (List[str]): An array representing a cipher text.
        createRelationsMatrix (bool): If True, creates a relations matrix at the end.

    Returns:
        Union[np.ndarray, List[np.ndarray]]:
            If createRelationsMatrix is True, returns a 2D numpy array representing the relations matrix.
            If createRelationsMatrix is False, returns a List of numpy arrays representing base vectors.
    """
    solvedMatrix = f.gaussianElimination(args, matrix)
    freeVariables = f.getFreeVariables(args, solvedMatrix)
    reducedMatrix = f.reduceMatrix(args, solvedMatrix, freeVariables)
    baseVectors = f.getBaseVectors(args, reducedMatrix, freeVariables)

    if createRelationsMatrix:
        relationsMatrix = f.calculateRelationsMatrix(args, baseVectors, relationsAmount, cipherText)
        return relationsMatrix
    else:
        return baseVectors


def executePipeline(args: Namespace) -> Tuple[List[np.ndarray], bool]:
    """
    Execute a pipeline to solve and verify the cryptographic problem.

    Returns:
        Tuple[List[np.ndarray], bool]: A tuple containing a list of a numpy array
                                       representing the base vector and a boolean
                                       indicating whether the verification of the
                                       result was successful.
    """
    # Execute the pipeline
    publicKey, cipherText, relationsAmount, matrix = generateMatrix(args)
    relationsMatrix = solveMatrix(args, matrix, relationsAmount, cipherText, True)
    baseVectors = solveMatrix(args, relationsMatrix, None, None, False)

    # Verify the result to ensure that the calculation was correct
    return baseVectors, f.verifyResult(args, publicKey, baseVectors, cipherText)
