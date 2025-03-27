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

import ray
from typing import Any, Iterator, Iterable, Union
from abc import ABC
from datasets import Dataset as HF_Dataset, IterableDataset as HF_IterableDataset


class Dataset(Iterable, ABC):
    """A base dataset class that serves as a base for other dataset
    implementations, including SFTDataset and TextCompletionDataset.
    It supports both regular and streaming Ray datasets.

    Args:
        source (Union[ray.data.Dataset, datasets.Dataset]): The underlying Ray dataset to wrap.

    """

    def __init__(self, source: Union[ray.data.Dataset, HF_Dataset]):
        self.source = self._maybe_convert_to_ray_dataset(source)
        self._length = None
        self._iterator = None

    def _maybe_convert_to_ray_dataset(self, source):
        # HuggingFace datasets
        if isinstance(source, (HF_Dataset, HF_IterableDataset)):
            return ray.data.from_huggingface(source)
        # TODO: Add adapters for other dataset formats
        return source

    def process_sample(self, sample):
        return sample

    def __len__(self) -> int:
        if self._length is None:
            try:
                print(
                    (
                        "Warning: If your dataset is a streaming dataset, "
                        "this operation (__len__) might trigger its lazy executation."
                    )
                )
                self._length = self.source.count()
            except:
                # For streamed datasets where count() might not work
                print(
                    (
                        "Unable to get __len__ of dataset. Dataset may be a streaming "
                        "dataset. Setting length to be 0."
                    )
                )
                self._length = 0
        return self._length

    def __next__(self) -> Any:
        """Returns the next element in the dataset."""
        if self._iterator is None:
            self._iterator = iter(self.source.iter_rows())
        sample = next(self._iterator)
        return self.process_sample(sample)

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over the dataset."""
        self._iterator = iter(self.source.iter_rows())
        return self
