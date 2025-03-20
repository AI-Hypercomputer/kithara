import jax

def f(x, y):
  return jax.numpy.sin(jax.numpy.cos(x+y)) * jax.numpy.sin(jax.numpy.cos(x-y))

primals, f_vjp = jax.vjp(f, 0.5, 1.0)
print("primals", primals)
print("f_vjp", type(f_vjp))

from kithara.distributed.sharding.utils import (
    entire_tree_is_sharded,
    is_not_sharded_and_is_large,
    get_size_in_mb,
    get_size_in_gb,
)
from kithara.utils.tree_utils import named_tree_map

def print_elements_that_are_unsharded_and_large_in_pytree(pytree):
    def print_fn(path, x):
        print("x", x)
        if is_not_sharded_and_is_large(x): 
            print(f"{path} is unsharded and has shape", x.shape)
    named_tree_map(print_fn, pytree)

print("f_vjp", f_vjp)



print_elements_that_are_unsharded_and_large_in_pytree(f_vjp)