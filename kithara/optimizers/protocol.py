from typing import (
    Protocol, Tuple, TypeAlias, List,
    runtime_checkable)

from jaxtyping import PyTree
from keras.src.backend import Variable

TrainableVariable: TypeAlias = List[Variable]
OptimizerVariable: TypeAlias = PyTree[Variable]

@runtime_checkable
class KitharaOptimizer(Protocol):

    @property
    def learning_rate(self) -> float | None:
        ...
    @property
    def variables(self) -> OptimizerVariable:
        ...
    @variables.setter
    def variables(self, state: PyTree):
        ...
    @property
    def iterations(self) -> Variable:
        ...
    def stateless_apply(self, optimizer_variable: OptimizerVariable, gradients: PyTree,
                        trainable_variables: TrainableVariable) -> Tuple[TrainableVariable, OptimizerVariable]:
        """Apply the optimizer to the trainable variables and gradients.

        Apply the optimizer to the trainable variables and gradients. This method should be stateless, i.e. it should not
        modify the internal state of the optimizer. If the optimizer has state, it should be passed as `optimizer_variable`.
        If `optimizer_variable` is not provided, the optimizer should use the internal state tracked by `KitharaOptimizer`,
        but not optimizer itself.

        Args:
            optimizer_variable (OptimizerVariable): The explicit optimizer state.
            gradients (PyTree): The gradients to apply to the trainable variables.
                The shape of the gradients should match the shape of the `trainable variables`.
            trainable_variables (TrainableVariable): The trainable variables to apply the optimizer to

        Returns:
            Tuple[TrainableVariable, OptimizerVariable]: The tuple of the updated trainable variables and the updated optimizer state.
        """
    def get_optimizer_memory_usage(self) -> int:
        """Get the memory usage of the state/variables of the optimizer in bytes.

        Returns:
            int: bytes used by the optimizer state/variables.
        """
