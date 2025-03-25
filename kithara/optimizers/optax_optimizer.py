from typing import Tuple

import jax
from jaxtyping import PyTree
from keras.src import backend
import optax

from kithara.optimizers.protocol import KitharaOptimizer, TrainableVariable, OptimizerVariable


class OptaxOptimizerInterface(KitharaOptimizer):
    def __init__(self, optimizer: optax.GradientTransformation, params: PyTree[backend.Variable]):
        self.optimizer, state = optimizer, optimizer.init(params)
        self.state_in_keras_variable = self._convert_state_to_keras_variables(state, params)

        with backend.name_scope("OptaxOptimizerInterface"):
            iterations = backend.Variable(0, name="iterations")

        self._iterations = iterations

    @property
    def learning_rate(self) -> None:
        return

    @property
    def variables(self) -> PyTree[backend.Variable]:
        return self.state_in_keras_variable

    @property
    def iterations(self) -> backend.Variable:
        return self._iterations

    def stateless_apply(self, optimizer_variable: OptimizerVariable, gradients: PyTree,
                        trainable_variables: TrainableVariable
                        ) -> Tuple[TrainableVariable, OptimizerVariable]:

        updates, opt_state = self.optimizer.update(
            gradients, optimizer_variable, trainable_variables)
        new_trainable_variables = optax.apply_updates(trainable_variables, updates)

        return new_trainable_variables, opt_state

    def get_optimizer_memory_usage(self):
        return sum([leaf.nbytes for leaf in jax.tree_util.tree_leaves(self.variables)])

    def _convert_state_to_keras_variables(
            self, opt_state: PyTree, trainable_vars: TrainableVariable) -> PyTree[backend.Variable]:
        """Convert optax State to a `PyTree` of `keras.Variable`

        Creates a structure (PyTree) of `keras.Variable` objects that mirrors
        the structure of the optimizer state (`opt_state`).

        For each subtree (e.g., momentum, variance) within the `opt_state` PyTree:
            - Corresponding `keras.Variable` objects are created, one for each
                associated trainable variable in the `trainable_vars` PyTree.
            - Each created `keras.Variable` is named based on the path of its associated node in `trainable_vars`:
                1. Take the trainable variable's path.
                2. Replace all forward slashes (`/`) with underscores (`_`).
                3. Append the suffix `_optstate`.

        For Example:
            - If a trainable variable's path in `trainable_vars` is
            `"decoder-block-0/layer-1/kernel"`.
            - The corresponding `keras.Variable` holding its optimizer state component
            will be named `"decoder-block-0_layer-1_kernel_optstate"`.

        Args:
            opt_state (PyTree): the optax optimizer state
            trainable_vars (PyTree[backend.Variable]): trainable variables

        Returns:
            PyTree[backend.Variable]: the transformed optax state.
        """

        _, trainable_tree_def = jax.tree.flatten(trainable_vars)

        def is_same_shape_with_trainable_subtree(opt_state_subtree: PyTree):
            _, opt_state_tree_def = jax.tree.flatten(opt_state_subtree)
            return opt_state_tree_def == trainable_tree_def

        def create_keras_variable_pytree(
                subtree, trainable_vars) -> PyTree[backend.Variable]:

            if not is_same_shape_with_trainable_subtree(subtree):
                return jax.tree.map(
                    lambda leaf: backend.Variable(leaf,), subtree
                )

            return jax.tree.map(
                lambda leaf, keras_var: backend.Variable(
                    leaf, name=f"{keras_var.path.replace('/', '_')}_optstate",),
                subtree,
                trainable_vars)

        return jax.tree.map(
            lambda subtree: create_keras_variable_pytree(subtree, trainable_vars),
            opt_state,
            is_leaf=is_same_shape_with_trainable_subtree
        )
