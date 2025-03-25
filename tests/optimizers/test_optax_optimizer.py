import unittest
import unittest.mock

import jax
import jax.numpy as jnp
import optax
from keras.src import backend

from kithara.optimizers import OptaxOptimizerInterface

class TestOptaxOptimizerInterface(unittest.TestCase):

    def setUp(self):
        self.dummy_keras_variables = [
            backend.Variable(jnp.ones((1024, 1024)), name="dummy_keras_variable")
            for _ in range(3)]

        return super().setUp()

    def test_learning_rate_is_none(self):
        optax_optimizer_interface = OptaxOptimizerInterface(
            optax.adam(1e-3), self.dummy_keras_variables)
        self.assertIsNone(optax_optimizer_interface.learning_rate)

    def test_optimizer_initialization(self):
        optax_optimizer_interface = OptaxOptimizerInterface(
            optax.adam(1e-3), self.dummy_keras_variables)
        self.assertTrue(
            jax.tree.reduce(
                lambda agg, leaf: agg and isinstance(leaf, backend.Variable),
                optax_optimizer_interface.variables, True))

    def test_stateless_apply(self):
        optax_optimizer_interface = OptaxOptimizerInterface(
            optax.adam(1e-3), self.dummy_keras_variables)
        new_trainables, new_opt_variables = optax_optimizer_interface.stateless_apply(
            optax_optimizer_interface.variables, self.dummy_keras_variables, self.dummy_keras_variables,)
        print(new_trainables, new_opt_variables)