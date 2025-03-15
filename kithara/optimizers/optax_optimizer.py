from typing import Tuple, Optional

import jax
from jaxtyping import PyTree
import optax

from kithara.optimizers.protocol import KitharaOptimizer, TrainableVariable, OptimizerVariable

class OptaxOptimizerInterface(KitharaOptimizer):
    def __init__(self, optimizer: optax.GradientTransformation, params: optax.Params):
        self.optimizer, self.state = optimizer, optimizer.init(params)

    @property
    def learning_rate(self) -> None:
        return

    @property
    def state_or_variables(self):
        return self.state

    @state_or_variables.setter
    def state_or_variables(self, state: PyTree):
        self.state = state

    def stateless_apply(self, trainable_variables: TrainableVariable, gradients: PyTree,
                        optimizer_variable: Optional[OptimizerVariable] = None,
                        ) -> Tuple[TrainableVariable, OptimizerVariable]:
        if optimizer_variable is not None:
            self.state_or_variables = optimizer_variable

        updates, self.state_or_variables = self.optimizer.update(gradients, self.state_or_variables,
                                                                 trainable_variables)
        new_trainable_variables = optax.apply_updates(trainable_variables, updates)
        return new_trainable_variables, self.state_or_variables

    def get_optimizer_memory_usage(self):
        return sum([leaf.nbytes for leaf in jax.tree_util.tree_leaves(self.state)])
