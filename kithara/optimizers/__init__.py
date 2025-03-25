from jaxtyping import PyTree
from kithara.optimizers.protocol import KitharaOptimizer
from kithara.optimizers.optax_optimizer import OptaxOptimizerInterface
import optax

def convert_to_kithara_optimizer(
        optimizer: optax.GradientTransformation, trainable_variables: PyTree) -> KitharaOptimizer:
    if isinstance(optimizer, optax.GradientTransformation):
        return OptaxOptimizerInterface(optimizer, trainable_variables)
    raise NotImplementedError(f"Unsupported optimizer type: {type(optimizer)}")
