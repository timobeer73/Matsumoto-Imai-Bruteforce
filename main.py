from pipeline import executePipeline
from argparse import ArgumentParser, Namespace


def setupArgumentParser() -> Namespace:
    parser = ArgumentParser(description='Decrypt a ciphertext of an Matsumoto-Imai-Encryption based on the given public key.')
    parser.add_argument('filepath', 
                        type=str, 
                        help='Path of the formatted file.')
    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Print additional information.')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = setupArgumentParser()
    executePipeline(args)
