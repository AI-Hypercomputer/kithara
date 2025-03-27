import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax
from kithara.distributed.sharding.utils import (
    entire_tree_is_sharded,
    is_not_sharded_and_is_large,
    get_size_in_mb,
)

class ValidationMixin:
    @staticmethod
    def _validate_sharding_correctness(
        data=None,
        model=None,
        optimizer=None,
    ):
        """This method performs several sharding correctness checks and prints
        warnings for any sharding issues detected.

        1. Checks if data is properly sharded
        2. Validates sharding of trainable variables
        3. Validates sharding of non-trainable variables
        4. Validates sharding of optimizer variables

        Args:
            data: Input batch to validate
            state: Current model state tuple

        """
        try:
            if data and not entire_tree_is_sharded(data):
                print(
                    "Warning: data is not sharded",
                    data["y"].shape,
                    data["y"].sharding,
                )
        except Exception as e:
            print(f"Error during sharding correctness validation: {e}")

        try:                    
            if optimizer:
                optimizer_variables = []
                for value in optimizer.variables:
                    if isinstance(value, keras.Variable):
                        optimizer_variables.append(value.value)
                    else:
                        optimizer_variables.append(value)

                _ = jax.tree.map(
                    lambda variable, value: (
                        print(
                            f"Optimizer variable is not sharded",
                            f"{get_size_in_mb(value)}mb",
                            variable.path,
                            value.shape,
                            value.sharding,
                        )
                        if is_not_sharded_and_is_large(value)
                        else None
                    ),
                    optimizer.variables,
                    optimizer_variables,
                )
        except Exception as e:
            print(
                f"Error during optimizer variable sharding correctness validation: {e}"
            )

        try:
    
            if model :
                model_variables = []
                for value in model.variables:
                    if isinstance(value, keras.Variable):
                        model_variables.append(value.value)
                    else:
                        model_variables.append(value)

                _ = jax.tree.map(
                    lambda variable, value: (
                        print(
                            f"Optimizer variable is not sharded",
                            f"{get_size_in_mb(value)}mb",
                            variable.path,
                            value.shape,
                            value.sharding,
                        )
                        if is_not_sharded_and_is_large(value)
                        else None
                    ),
                    model.variables,
                    model_variables,
                )
        except Exception as e:
            print(f"Error during model variable sharding correctness validation: {e}")
    
    @staticmethod
    def _validate_memory_usage(models, optimizers):
        """This method checks the current HBM usage matches the expected HBM
        usage.

        Current HBM usage is calculated by summing the size of all live arrays,
        expected HBM usage is calculated by summing the size of all model and
        optimizer variables.
        """

        total_size = 0
        for model in models:
            for v in model.variables:
                total_size += get_size_in_mb(v.value)
        for optimizer in optimizers:
            total_size += jax.tree.reduce(
                lambda agg, leaf: jax.numpy.add(agg, get_size_in_mb(leaf.value)),
                optimizer.variables,
                initializer=0,
            )

        live_arrays = jax.live_arrays()
        live_arrays_size = 0
        for v in live_arrays:
            live_arrays_size += get_size_in_mb(v)

        if not np.isclose(total_size, live_arrays_size, atol=1.0):
            print(
                f"WARNING: Potential memory leakage. HBM usage is {live_arrays_size:.3f} MB "
                f"but model and optimizer are only {total_size:.3f} MB in size."
            )
        else:
            print(
                f"✅ No memory leakage detected. HBM usage ({live_arrays_size:.3f} MB) "
                f"matches model and optimizer size ({total_size:.3f} MB)."
            )

        try:
            memory_info = jax.local_devices()[0].memory_stats()
            memory_per_device_mb = memory_info["bytes_limit"] / (1024**2)
            total_memory = memory_per_device_mb * jax.device_count()
            print(
                f"Total memory available is {total_memory:.3f} MB, if you run into "
                "errors, check if your memory usage is close to the limit, and either "
                "reduce your per-device batch size or sequence length."
            )
        except Exception as e:
            # memory_info is not available on some TPUs
            pass
