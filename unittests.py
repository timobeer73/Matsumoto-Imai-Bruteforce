import main
import numpy
import unittest
import testVariables as v


class TestSumArray(unittest.TestCase):
    def testReadFilePublicKey(self):
        self.assertIn(v.publicKey,
                      main.readFile(v.args))


    def testReadFileCipherText(self):
        self.assertIn(v.cipherText,
                      main.readFile(v.args))


    def testReadFileRelationsAmount(self):
        self.assertIn(v.relationsAmount, 
                      main.readFile(v.args))
        
        
    def testGeneratePlaintextLength(self):
        expectedLength = 2 * pow(v.relationsAmount, 2)
        self.assertEqual(expectedLength, 
                         len(main.generatePlainText(v.args, 
                                                    v.relationsAmount)))
        

    def testCalculateCipherTextMatrix(self):
        self.assertTrue(numpy.array_equal(v.correspondingCipherTextMatrix,
                                          main.calculateCipherText(v.args, 
                                                                   v.publicKey, 
                                                                   v.plainTextMatrix)))
        
        
    def testCalculateMatrix(self):
        self.assertTrue(numpy.array_equal(v.correspondingMatrix,
                                          main.calculateMatrix(v.args, 
                                                               v.plainTextMatrix, 
                                                               v.correspondingCipherTextMatrix)))
        
        
    def testGaussianElimination(self):
        self.assertTrue(numpy.array_equal(v.solvedMatrix,
                                          main.gaussianElimination(v.args, 
                                                                   v.unsolvedMatrix)))
        
    
    def testGetFreeVariables(self):
        self.assertEqual([3, 4], 
                         main.getFreeVariables(v.args, v.solvedMatrix))
        

if __name__ == '__main__':
    unittest.main()
