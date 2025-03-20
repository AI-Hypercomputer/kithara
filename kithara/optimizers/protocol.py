from typing import (
    Protocol, Tuple,
    TypeAlias, Optional,
    runtime_checkable)
from jaxtyping import PyTree, Array

TrainableVariable: TypeAlias = PyTree[Array]
OptimizerVariable: TypeAlias = PyTree[Array]

@runtime_checkable
class KitharaOptimizer(Protocol):

    @property
    def learning_rate(self) -> float | None:
        ...
    @property
    def state_or_variables(self) -> PyTree:
        ...
    @state_or_variables.setter
    def state_or_variables(self, state: PyTree):
        ...
    def stateless_apply(self, trainable_variables: TrainableVariable, gradients: PyTree,
                        optimizer_variable: Optional[OptimizerVariable] = None,) -> Tuple[TrainableVariable, OptimizerVariable]:
        """Apply the optimizer to the trainable variables and gradients.

        Apply the optimizer to the trainable variables and gradients. This method should be stateless, i.e. it should not
        modify the internal state of the optimizer. If the optimizer has state, it should be passed as `optimizer_variable`.
        If `optimizer_variable` is not provided, the optimizer should use the internal state tracked by `KitharaOptimizer`,
        but not optimizer itself.

        Args:
            trainable_variables (TrainableVariable): The trainable variables to apply the optimizer to
            gradients (PyTree): The gradients to apply to the trainable variables.
                The shape of the gradients should match the shape of the `trainable variables`.
            optimizer_variable (Optional[OptimizerVariable], optional): The explicit optimizer state. Defaults to None.

        Returns:
            Tuple[TrainableVariable, OptimizerVariable]: The tuple of the updated trainable variables and the updated optimizer state.
        """
    def get_optimizer_memory_usage(self) -> int:
        """Get the memory usage of the state/variables of the optimizer in bytes.

        Returns:
            int: bytes used by the optimizer state/variables.
        """
