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

from kithara.distributed.sharding.models import GEMMA_LAYOUT
from dataclasses import dataclass
from typing import ClassVar


# Layout configurations for different model architectures
@dataclass
class Layout:
    # Class-level dictionary to store mesh types
    _mesh_types: ClassVar[dict] = {
        "gemma": lambda: Layout.gemma(),
    }

    def __class_getitem__(cls, key: str):
        if key not in cls._mesh_types:
            raise KeyError(f"Unknown mesh type: {key}")
        return cls._mesh_types[key]()

    @classmethod
    def gemma(cls):
        return GEMMA_LAYOUT
