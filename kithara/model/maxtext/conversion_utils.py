"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""
This module provides functionality to converting a MaxText model (Flax model) to a Keras model.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras

from typing import Any, Dict, List, Union, Optional
import flax
from functools import lru_cache
import jax
from kithara.utils.tree_utils import named_tree_map
from keras.src.utils import tracking
from keras.src import backend
import numpy as np
from keras.layers import Input
from keras.src.utils.jax_layer import FlaxLayer
import functools
from kithara.model import set_global_sharding_strategy
from kithara.distributed.sharding import ShardingStrategy
import jax.numpy as jnp
import time

class MaxTextLayer(FlaxLayer):

    @tracking.no_automatic_dependency_tracking
    def _create_variables(self, values: Dict[str, Any], trainable: bool) -> List[Any]:
        """Overrite FlaxLayer's _create_variables so that a KerasVariable is created
        with the name argument set to be the path prefix of it's parents in the pytree.

        Args:
            values: Dictionary containing the variable values
            trainable: Whether the variables should be trainable

        Returns:
            List of created variables
        """

        def create_variable(name: List[str], value: Any) -> Any:
            name_str = "-".join(name)
            if backend.is_tensor(value) or isinstance(value, np.ndarray):
                variable = self.add_weight(
                    value.shape, initializer=value, trainable=trainable, name=name_str
                )
                return variable
            elif isinstance(value, (np.generic, int, float)):
                variable = self.add_weight(
                    (), initializer=value, trainable=trainable, name=name_str
                )
                return variable
            else:
                return value

        # Use JAX's tree_map as it understands registered classes.
        variables = named_tree_map(create_variable, values)

        if trainable:
            self.params = variables
        else:
            self.state = variables

        flat_variables, _ = jax.tree_util.tree_flatten(variables)
        return flat_variables


class MaxTextConversionMixin:

    @staticmethod
    def convert_maxtext_model_to_keras_model(
        maxtext_model, state, seq_len: int, global_batch_size: int, mesh, config
    ) -> keras.Model:
        """Convert a MaxText model to a Keras model.

        This utility function converts a MaxText (Flax) model into a Keras Model instance.
        It creates a thin translation layer where the weights of the Flax model are registered
        as Keras variables. During the Keras model's call(), the computations are forwarded
        to the Flax model's apply() method.

            maxtext_model (flax.linen.Module): The MaxText model to convert. This model
                should be obtained from `MaxTextConversionMixin.initialize_random_model()`.
            state: The state of the Flax model.
            seq_len (int): Length of the input sequence.
            global_batch_size (int): Batch size for the model.
            mesh: The mesh configuration for partitioning.
            config: Configuration object containing logical axis rules.

        Returns:
            A Keras Model instance
        """

        def maxtext_wrapper(
            module: Any, inputs: List[Union[np.ndarray, jnp.ndarray]], training: bool
        ) -> Any:
            print("inside maxtext wrapper")
            tokens, positions, segment_ids = inputs
            model_mode = "train"
            segment_ids = segment_ids if training else None
            with mesh, flax.linen.partitioning.axis_rules(config.logical_axis_rules):
                print("running forward call")
                return module(
                    tokens,
                    positions,
                    segment_ids,
                    enable_dropout=training,
                    model_mode=model_mode,
                )
        print(f"-> Converting the MaxText model into a Keras Model...")
        keras_layer = MaxTextLayer(
            module=maxtext_model, method=maxtext_wrapper, variables=state
        )

        # Build the Keras model
        tokens = Input(
            shape=(seq_len,), batch_size=global_batch_size, dtype="int32", name="tokens"
        )
        positions = Input(
            shape=(seq_len,),
            batch_size=global_batch_size,
            dtype="int32",
            name="positions",
        )
        segment_ids = Input(
            shape=(seq_len,),
            batch_size=global_batch_size,
            dtype="int32",
            name="segment_ids",
        )
        logits = keras_layer([tokens, positions, segment_ids], training=True)
        keras_model = keras.Model(inputs=[tokens, positions, segment_ids], outputs=logits)
        return keras_model

    @staticmethod
    @lru_cache(maxsize=1)
    def get_maxtext_pyconfig(
        model_name: Optional[str] = None, maxtext_config: Optional[str] = None
    ):
        from kithara.model.maxtext.maxtext.MaxText import pyconfig
        base_yaml_file = os.path.join(os.path.dirname(__file__), "maxtext", "MaxText", "configs", "base.yml")

        argv = [
            "",
            base_yaml_file,
            "run_name=must_supply_but_not_needed",
            "skip_jax_distributed_system=True",
        ]

        if model_name is not None:
            argv += [f"model_name={model_name}"]
        if maxtext_config is not None:
            argv += maxtext_config.split(" ")
        config = pyconfig.initialize(argv)
        return config

    @staticmethod
    def initialize_random_maxtext_model(
        model_name: str,
        seq_len: int,
        per_device_batch_size: int,
        weight_dtype: str,
        activation_dtype: str,
        scan_layers: bool,
        max_prefill_predict_length: int,
        maxtext_config_args: Optional[str] = None,
    ) -> tuple[ShardingStrategy, keras.Model]:
        """Initialize a random MaxText model with the input configuration.

        This internal method handles the low-level initialization of the model,
        including mesh setup, parameter initialization, and conversion to Keras format.

        Args:
            model_name (str): Name of the model configuration.
            seq_len (int): Maximum sequence length.
            per_device_batch_size (int): Batch size per device.
            weight_dtype (str): Data type for model weights.
            activation_dtype (str): Data type for activations.
            scan_layers (bool): Whether to use scan layers.
            maxtext_config_args (Optional[str], optional): Additional configuration arguments. Defaults to None.

        Returns:
            tuple[ShardingStrategy, keras.Model]: Tuple containing sharding strategy and initialized model.
        """

        print(f"-> Initializing a MaxText {model_name} model...")
        start_time = time.time()
        
        from kithara.distributed.sharding.maxtext import MaxTextSharding
        from kithara.model.maxtext.maxtext.MaxText.train import setup_mesh_and_model
        from kithara.model.maxtext.maxtext.MaxText.max_utils import (
            get_abstract_state,
            unbox_logicallypartioned,
        )

        if maxtext_config_args == None:
            maxtext_config_args = {}

        assert "weight_dtype" not in maxtext_config_args
        maxtext_config_args["weight_dtype"] = weight_dtype
        assert "dtype" not in maxtext_config_args
        maxtext_config_args["dtype"] = activation_dtype
        assert "scan_layers" not in maxtext_config_args
        maxtext_config_args["scan_layers"] = scan_layers
        assert "per_device_batch_size" not in maxtext_config_args
        maxtext_config_args["per_device_batch_size"] = per_device_batch_size
        assert "max_target_length" not in maxtext_config_args
        maxtext_config_args["max_target_length"] = seq_len
        assert "max_prefill_predict_length" not in maxtext_config_args
        maxtext_config_args["max_prefill_predict_length"] = max_prefill_predict_length
        if "enable_checkpointing" not in maxtext_config_args:
            maxtext_config_args["enable_checkpointing"] = False
        
        maxtext_config_args = " ".join(
            [f"{key}={value}" for key, value in maxtext_config_args.items()]
        )

        maxtext_config = MaxTextConversionMixin.get_maxtext_pyconfig(
            model_name, maxtext_config_args
        )
        global_batch_size = per_device_batch_size * jax.device_count()

        # Initialize the model and mesh configuration
        init_rng, _, _, jax_mesh, model, _, tx = setup_mesh_and_model(maxtext_config)

        # Initialize model parameters
        def init_initial_state(model, rng):
            input_shape = (global_batch_size, seq_len)
            return model.init(
                {"params": rng, "dropout": rng, "aqt": rng},
                np.ones(input_shape, dtype=jnp.int32),
                np.ones(input_shape, dtype=jnp.int32),
            )

        init_state_partial = functools.partial(init_initial_state, model)

        # Get the model state and shardings
        _, _, state_shardings = get_abstract_state(
            model, tx, maxtext_config, init_rng, jax_mesh, is_training=True
        )
        state = jax.jit(
            init_state_partial, in_shardings=None, out_shardings=state_shardings.params
        )(init_rng)
        state = unbox_logicallypartioned(state)

        sharding_strategy = MaxTextSharding(jax_mesh, state_shardings, maxtext_config)
        set_global_sharding_strategy(sharding_strategy)

        # Convert to Keras model format
        model = MaxTextConversionMixin.convert_maxtext_model_to_keras_model(
            model, state, seq_len, global_batch_size, jax_mesh, maxtext_config
        )

        print(f"✅ Successfully initialized a MaxText {model_name} model in {time.time() - start_time:.3f}s...")
        return maxtext_config, sharding_strategy, model

    def get_maxtext_params(model: "kithara.MaxTextModel") -> dict:
        """Convert Kithara variables (flat list format) into 
        MaxText variables (nested dict format). This function 
        is a simple format converter. It is used for inserting 
        the current model parameters into MaxText's inference 
        engine. 
        """
        params = {}
        for v in model.variables:
            variable_name = v.path
            nest_keys = variable_name.split("/")[-1].split("-")
            if nest_keys[0] != "params":
                continue
                
            current = params
            for key in nest_keys[:-1]:  # All keys except the last one
                current = current.setdefault(key, {})
            current[nest_keys[-1]] = v.value
            
        return params

    def get_maxtext_params_from_variables(variables: "kithara.MaxTextModel") -> dict:
        """Convert Kithara variables (flat list format) into 
        MaxText variables (nested dict format). This function 
        is a simple format converter. It is used for inserting 
        the current model parameters into MaxText's inference 
        engine. 
        """
        params = {}
        for v in variables:
            variable_name = v.path
            nest_keys = variable_name.split("/")[-1].split("-")
            if nest_keys[0] != "params":
                continue
                
            current = params
            for key in nest_keys[:-1]:  # All keys except the last one
                current = current.setdefault(key, {})
            current[nest_keys[-1]] = v.value
            
        return params
