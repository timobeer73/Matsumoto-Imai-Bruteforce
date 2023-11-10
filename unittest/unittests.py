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


    def testReadFileRelations(self):
        self.assertIn(v.relationsAmount, 
                      main.readFile(v.args))
        
        
    def testGeneratePlaintext(self):
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
        
        
    def testReduceMatrix(self):
        self.assertTrue(numpy.array_equal(v.reducedMatrix, 
                        main.reduceMatrix(v.args, 
                                          v.unreducedMatrix, 
                                          [5, 6, 8])))
        
    
    def testGetBaseVectors(self):
        self.assertTrue(numpy.array_equal(v.correspondingBaseVectors, 
                        main.getBaseVectors(v.args, 
                                            v.reducedMatrix, 
                                            [5, 6, 8])))
        

if __name__ == '__main__':
    unittest.main()
