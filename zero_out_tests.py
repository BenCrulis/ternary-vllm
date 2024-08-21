import os

import tensorflow as tf

from custom_ops_mod import zero_out

class ZeroOutTest():
    def testZeroOut(self):
        result = zero_out([[5, 4, 3, 2, 1], [10,9,8,7,6]])
        assert tf.reduce_all(result.numpy() == [5, 0, 0, 0, 0])


if __name__ == "__main__":
    # print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])
    # print("PYTHONPATH:", os.environ["PYTHONPATH"])
    testCase = ZeroOutTest()
    testCase.testZeroOut()
