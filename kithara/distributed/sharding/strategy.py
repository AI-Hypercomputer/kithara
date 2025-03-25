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

from dataclasses import dataclass
from typing import Optional
from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel
from keras.src.distribution.distribution_lib import Distribution
from jax.sharding import Sharding
from kithara.distributed.sharding._mesh import Mesh as PredefinedMesh
from kithara.distributed.sharding._layout import Layout as PredefinedLayout
from kithara.distributed.sharding._data_sharding import DataSharding
from abc import ABC, abstractmethod
from keras.src.backend.common import global_state

@dataclass
class ShardingStrategy(ABC):
    """Abstract base class for Kithara sharding strategies.

    A sharding strategy defines how to shard model and optimizer parameters,
    and input data across all devices.

    Properties:
        mesh: The device mesh configuration for distributed training
        layout_map: The layout mapping for model parameters across devices
        data_sharding: The sharding specification for input data
        distribution: The Keras distribution strategy

    See `PredefinedShardingStrategy` for an example.
    """

    @property
    @abstractmethod
    def mesh(self) -> DeviceMesh:
        pass

    @property
    @abstractmethod
    def layout_map(self) -> LayoutMap:
        pass

    @property
    @abstractmethod
    def data_sharding(self) -> Sharding:
        pass

    @property
    @abstractmethod
    def distribution(self) -> Distribution:
        pass

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the sharding strategy configuration."""
        if not isinstance(self.mesh, DeviceMesh):
            raise ValueError(
                f"mesh must be an instance of keras.distribution.DeviceMesh but is {self.mesh}"
            )
        if not isinstance(self.layout_map, LayoutMap):
            raise ValueError(
                f"layout_map must be an instance of keras.distribution.LayoutMap but is {self.layout_map}"
            )
        if not isinstance(self.data_sharding, Sharding):
            raise ValueError(
                f"data_sharding must be an instance of jax.sharding.Sharding but is {self.data_sharding}"
            )
        if not isinstance(self.distribution, Distribution):
            raise ValueError(
                f"distribution must be an instance of keras.distribution.Distribution but is {self.distribution}"
            )

@dataclass
class PredefinedShardingStrategy(ShardingStrategy):
    """This class provides pre-configured sharding strategies optimized for specific
    model architectures and parallelism types.

    Args:
        parallelism (str): The type of parallelism to use. Must be one of ["fsdp", "tp"]
        model (str): The model architecture. Must be one of ["gemma"]

    Example:
        ```python
        strategy = PredefinedShardingStrategy(
            parallelism="fsdp",
            model="gemma2-27b"
        )
        set_global_sharding_strategy(strategy)
        ```
    """

    parallelism: str
    model: str

    @property
    def mesh(self) -> DeviceMesh:
        return PredefinedMesh[self.parallelism]

    @property
    def layout_map(self) -> LayoutMap:
        layout_dict = PredefinedLayout[self.model][self.parallelism]
        layout_map = LayoutMap(self.mesh)
        for pattern, layout in layout_dict.items():
            layout_map[pattern] = layout
        return layout_map

    @property
    def data_sharding(self) -> Sharding:
        return DataSharding[self.parallelism]

    @property
    def distribution(self) -> Distribution:
        return ModelParallel(layout_map=self.layout_map)

