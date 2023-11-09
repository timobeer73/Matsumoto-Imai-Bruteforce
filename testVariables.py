import numpy
from argparse import Namespace


# Correct values from the test file for comparison
args = Namespace(filepath='testFile.txt', 
                 verbose=False)
publicKey = ['x_1*x_2+x_2*x_3+x_2*x_4+x_3*x_4+x_3*x_5', 
             'x_1*x_2+x_2*x_5+x_4*x_5+x_2+x_3+x_4', 
             'x_2*x_3+x_3*x_4+x_1+x_2+x_4', 
             'x_1*x_2+x_2*x_3+x_2+x_3+x_4', 
             'x_1*x_4+x_3*x_4+x_2*x_5+x_3*x_5+x_4*x_5+x_1+x_2+x_3+x_4']
cipherText = ['1', '0', '1', '1', '0']
relationsAmount = 5

# Example values for comparison
plainTextMatrix = numpy.array([[False, False, False, False, False],
                               [True,  False, False, False, False],
                               [False, True,  False, False, False],
                               [False, False, True,  False, False],
                               [False, False, False, True,  False],
                               [False, False, False, False, True ],
                               [True,  True,  True,  True,  True ]])
correspondingCipherTextMatrix = numpy.array([[False, False, False, False, False],
                                             [False, False, True,  False, True ],
                                             [False, True,  True,  True,  True ],
                                             [False, True,  False, True,  True ],
                                             [False, True,  True,  True,  True ],
                                             [False, False, False, False, False],
                                             [True,  False, True,  True,  True ]])
correspondingMatrix = numpy.array([[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                                   [False, False, True,  False, True,  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                                   [False, False, False, False, False, False, True,  True,  True,  True,  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                                   [False, False, False, False, False, False, False, False, False, False, False, True,  False, True,  True,  False, False, False, False, False, False, False, False, False, False],
                                   [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True,  True,  True,  True,  False, False, False, False, False],
                                   [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                                   [True,  False, True,  True,  True,  True, False,  True,  True,  True,  True,  False, True,  True,  True,  True,  False, True,  True,  True,  True,  False, True,  True,  True]])
unsolvedMatrix = numpy.array([[True,  False, True,  False, True ],
                              [False, True,  False, True,  False],
                              [True,  True,  False, True,  True ],
                              [False, False, True,  False, False],
                              [True,  True,  True,  True,  True ],
                              [False, False, False, False, False],
                              [True,  False, True,  False, True ],
                              [False, True,  False, True,  False],
                              [True,  True,  False, True,  True ],
                              [False, False, True,  False, False]])
solvedMatrix = numpy.array([[True,  True,  True,  True,  True ],
                            [False, True,  False, True,  False],
                            [False, False, True,  False, False]])
