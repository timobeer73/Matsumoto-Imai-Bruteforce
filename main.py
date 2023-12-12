import numpy as np
from time import time
from datetime import datetime
from pipeline import executePipeline
from argparse import ArgumentParser, Namespace


def setupArgumentParser() -> Namespace:
    """
    Set up and configure the argument parser for the cryptographic decryption script.

    Returns:
        Namespace: A namespace containing the parsed command-line arguments.
    """
    parser = ArgumentParser(description='Decrypt a ciphertext using Matsumoto-Imai-Encryption based on the given public key.')
    parser.add_argument('filepath',
                        type=str,
                        help='Path of the formatted file.')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Print additional information.')

    return parser.parse_args()


if __name__ == '__main__':
    startingTime = time()
    
    args = setupArgumentParser()    
    baseVectors, isCorrect = executePipeline(args)
        
    currentTime = datetime.now().strftime("%H:%M:%S")  
    if isCorrect:
        print(f'[{currentTime}] Successful! Plain text solution: {np.array(baseVectors, dtype=np.uint8)}')
        if args.verbose:
              print(f'[{currentTime}] Solved in: {round((time() - startingTime), 2)} seconds')
    else:
        print(f'[{currentTime}] Decryption failed')
