import numpy as np
import functions as f
from typing import List, Tuple
from argparse import Namespace


def calculateInitialMatrix(args: Namespace) -> Tuple[List[str], List[str], int, np.ndarray]:
    """
    Calculate the initial matrices and values required for the cryptographic problem.

    Returns:
        Tuple[List[str], List[str], int, np.ndarray]: A tuple containing the public key,
                                                      cipher text, relationsAmount and the 
                                                      initial matrix.
    """
    publicKey, cipherText, relationsAmount = f.readFile(args)
    plainTextArray = f.generatePlainText(args, relationsAmount)
    cipherTextsArray = f.calculateCipherText(args, publicKey, plainTextArray)
    matrix = f.calculateMatrix(args, plainTextArray, cipherTextsArray)
    
    return publicKey, cipherText, relationsAmount, matrix


def solveInitialMatrix(args: Namespace, matrix: np.ndarray, relationsAmount: int, cipherText: List[str]) -> np.ndarray:
    """
    Solve the initial matrix and obtain the relations matrix.

    Args:
        matrix (np.ndarray): The initial matrix.
        relationsAmount (int): The number of basic elements/special relations.
        cipherText (List[str]): An array representing a cipher text.

    Returns:
        np.ndarray: A 2D numpy array representing the relations matrix.
    """
    solvedMatrix = f.gaussianElimination(args, matrix)
    freeVariables = f.getFreeVariables(args, solvedMatrix)
    reducedMatrix = f.reduceMatrix(args, solvedMatrix, freeVariables)
    baseVectors = f.getBaseVectors(args, reducedMatrix, freeVariables)
    relationsMatrix = f.calculateRelationsMatrix(args, baseVectors, relationsAmount, cipherText)
    
    return relationsMatrix


def solveRelationsMatrix(args: Namespace, relationsMatrix: np.ndarray) -> List[np.ndarray]:
    """
    Solve the relations matrix and obtain the base vectors (plain text of the intercepted
    cipher text).

    Args:
        relationsMatrix (np.ndarray): A 2D numpy array representing the relations matrix.

    Returns:
        List[np.ndarray]: A list of a numpy array representing the base vector.
    """
    solvedMatrix = f.gaussianElimination(args, relationsMatrix)
    freeVariables = f.getFreeVariables(args, solvedMatrix)
    reducedMatrix = f.reduceMatrix(args, solvedMatrix, freeVariables)
    baseVectors = f.getBaseVectors(args, reducedMatrix, freeVariables)
    
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
    publicKey, cipherText, relationsAmount, matrix = calculateInitialMatrix(args)
    relationsMatrix = solveInitialMatrix(args, matrix, relationsAmount, cipherText)
    baseVectors = solveRelationsMatrix(args, relationsMatrix)

    # Verify the result to insure that the calculation was right
    return baseVectors, f.verifyResult(args, publicKey, baseVectors, cipherText)
