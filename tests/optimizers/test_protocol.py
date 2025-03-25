"""Unit tests for testing the protocols integrity and the conversion of optimizers to Kithara optimizers.

Run test on a TPU VM: python -m unittest tests/optimizers/test_protocol.py
"""
import unittest
import unittest.mock

import keras
import optax

from kithara.optimizers import KitharaOptimizer, convert_to_kithara_optimizer

class TestKitharaOptimizer(unittest.TestCase):

    def test_kithara_optimizer_interface_integrity(self):

        with self.assertRaises(TypeError):
            KitharaOptimizer()
        self.assertTrue(callable(KitharaOptimizer.stateless_apply))
        self.assertTrue(callable(KitharaOptimizer.get_optimizer_memory_usage))
        self.assertTrue(hasattr(KitharaOptimizer, "learning_rate"))
        self.assertTrue(hasattr(KitharaOptimizer, "iterations"))
        self.assertTrue(hasattr(KitharaOptimizer, "variables"))

    @unittest.mock.patch("kithara.optimizers.OptaxOptimizerInterface")
    def test_convert_to_kithara_optimizer(
        self, mocked_optax_optimizer_interface: unittest.mock.Mock):

        convert_to_kithara_optimizer(optax.adam(1e-3), 1)
        mocked_optax_optimizer_interface.assert_called_once()

        with self.assertRaises(NotImplementedError):
            convert_to_kithara_optimizer(keras.Optimizer(.01), 1.0)

if __name__ == "__main__":
    unittest.main()