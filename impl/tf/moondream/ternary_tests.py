import unittest
from unittest import TestCase

import tensorflow as tf
# import sys
# import os
# sys.path += ["../../.."]
from ternary import pack_ternary, unpack_ternary_iterative, pack_ternary2, unpack_ternary_iterative2


class TernaryCase(TestCase):
    def __init__(self, pack_fn=pack_ternary, unpack_fn=unpack_ternary_iterative, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.pack_fn = pack_fn
        self.unpack_fn = unpack_fn

    def test_packing_unpacking(self):
        
        t = tf.random.uniform((5031,), -1, 1, dtype=tf.int32)

        packed = self.pack_fn(t)
        unpacked = self.unpack_fn(packed)
        unpacked = tf.cast(unpacked, dtype=t.dtype)[:t.shape[-1]]

        self.assertTrue(tf.reduce_all(t == unpacked))
        pass


if __name__ == '__main__':
    print("testing")
    import sys
    sys.argv = sys.argv[:1]

    testCase = TernaryCase()
    testCase.test_packing_unpacking()

    testCase = TernaryCase(pack_fn=pack_ternary2, unpack_fn=unpack_ternary_iterative2)
    testCase.test_packing_unpacking()

    # unittest.main()