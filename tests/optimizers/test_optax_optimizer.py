import unittest
import unittest.mock

import jax.numpy as jnp
import optax

from kithara.optimizers import OptaxOptimizerInterface

class TestOptaxOptimizerInterface(unittest.TestCase):

    @unittest.mock.patch("optax.GradientTransformation.init")
    def test_learning_rate_is_none(self, _):
        optax_optimizer_interface = OptaxOptimizerInterface(optax.adam(1e-3), 1)
        self.assertIsNone(optax_optimizer_interface.learning_rate)

    @unittest.mock.patch("optax.GradientTransformation.init")
    def test_set_state_or_variable(self, _):
        optax_optimizer_interface = OptaxOptimizerInterface(optax.adam(1e-3), 1)
        optax_optimizer_interface.state_or_variables = 2
        self.assertEqual(optax_optimizer_interface.state_or_variables, 2)

    @unittest.mock.patch("optax.GradientTransformation.init")
    @unittest.mock.patch("optax.GradientTransformation.update",
                         return_value=(1, 1))
    @unittest.mock.patch("optax.apply_updates")
    def test_stateless_apply(
        self, mocked_apply_updates: unittest.mock.Mock,
        mocked_update: unittest.mock.Mock, _
    ):
        optax_optimizer_interface = OptaxOptimizerInterface(optax.adam(1e-3), 1)
        optax_optimizer_interface.stateless_apply(3, 4)
        mocked_update.assert_called_once()
        mocked_apply_updates.assert_called_once()

    @unittest.mock.patch("optax.GradientTransformation.init",
                         return_value=([jnp.ones((1024, 1024)) for _ in range(3)]))
    def test_get_optimizer_memory_usage(self, _):
        optax_optimizer_interface = OptaxOptimizerInterface(optax.adam(1e-3), 1)
        self.assertEqual(optax_optimizer_interface.get_optimizer_memory_usage(), 3 * 1024 * 1024 * 4)