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

from ctypes import cdll
import subprocess
import shutil
from keras.src.callbacks.callback import Callback
import jax
import os

class Profiler(Callback):
    """A flexible profiling callback for Keras models that supports both JAX XLA and NVIDIA NSys profiling.
    
    This class can be used either as a Keras callback during model training or as a standalone
    profiler through manual activation/deactivation. It supports two profiling modes:
    - 'xplane': Uses JAX's built-in profiler to collect XLA/TPU execution data
    - 'nsys': Uses NVIDIA's NSys profiler to collect GPU execution data
    
    When used as a Keras callback, the profiler automatically starts after skipping a specified
    number of steps and runs for a specified number of steps. When used standalone, profiling
    can be manually controlled using activate() and deactivate() methods.
    
    Args:
        output_path (str, optional): Directory path where profiling results will be saved.
            Defaults to "profiler".
        mode (str, optional): Profiling mode, either "xplane" for JAX/TPU profiling or "nsys" 
            for NVIDIA GPU profiling. Defaults to "xplane".
        max_profile_steps (int, optional): Number of training steps to profile when used as 
            a callback. Defaults to 5.
        skip_first_n_steps (int, optional): Number of initial training steps to skip before 
            starting profiling when used as a callback. This helps avoid profiling 
            initialization overhead. Defaults to 5.
        upload_all_profiler_results (bool, optional): If True, saves profiling results from 
            all processes. If False, only saves results from process 0. Defaults to False.
        optional_postfix (str, optional): String to append to the output directory path.
            Defaults to "".
                    
    Examples:
        Using as a Keras callback:
        ```python
        profiler = Profiler(output_path="./profiles", mode="xplane")
        model.fit(x_train, y_train, callbacks=[profiler])
        ```
        
        Using as a standalone profiler:
        ```python
        profiler = Profiler(mode="nsys")
        profiler.activate()
        # ... code to profile ...
        profiler.deactivate()
        ```
    
    Note: The nsys mode is currently not well tested. 
    """

    def __init__(
        self,
        output_path="profiler",
        mode="xplane",
        max_profile_steps=5,
        skip_first_n_steps=5,
        upload_all_profiler_results: bool = False,
        optional_postfix=""
    ):
        super().__init__()
        self.libcudart = None
        self.output_path = os.path.join(output_path, optional_postfix)
        self.max_profile_steps = max_profile_steps
        self.skip_first_n_steps = skip_first_n_steps
        self.mode = mode
        self.upload_all_profiler_results = upload_all_profiler_results
        self._is_tracing = False 

        if mode not in ["nsys", "xplane"]:
            raise ValueError("Profiler mode is not supported. Supported modes are xplane and nsys.")

    def on_train_begin(self, logs=None):
        self._global_train_batch = 0

    def on_train_end(self, logs=None):
        if self._is_tracing:
            self.deactivate()

    def on_train_batch_begin(self, batch, logs=None):
        self._global_train_batch += 1
        if self._global_train_batch == self.skip_first_n_steps:
            self.activate()

    def on_train_batch_end(self, batch, logs=None):

        if self._is_tracing and (self._global_train_batch == self.skip_first_n_steps + self.max_profile_steps):
            self.deactivate()

    def activate(self):
        """Start the profiler.
        nsys profiler becomes no-op when libcudart.so is not available on the system"""
        if not (self.upload_all_profiler_results or jax.process_index() == 0):
            return
        if self.mode == "nsys":
            try:
                self.libcudart = cdll.LoadLibrary("libcudart.so")
            except Exception as e:
                print(
                    f"WARNING: Failed to load library for nsys: {e}\n" "profiler has no effect")
                return
            self.libcudart.cudaProfilerStart()
        elif self.mode == "xplane":
            jax.profiler.start_trace(self.output_path)
        else: 
            raise ValueError("Profiler mode is not supported. Supported modes are xplane and nsys.")
        self._is_tracing = True
        print(f"Profiler started tracing. Profile will be saved to {self.output_path}")


    def deactivate(self):
        """End the profiler.
        The result is uploaded to the output bucket"""
        if not (self.upload_all_profiler_results or jax.process_index() == 0):
            return
        if self.mode == "nsys":
            if self.libcudart is not None:
                self.libcudart.cudaProfilerStop()
            else:
                print(
                    "WARNING: library for nsys was not loaded \n" "profiler has no effect")
                return
            # Popen() instead of run() for non-blocking behavior
            if shutil.which("gcloud") is not None:
                subprocess.Popen(
                    ["gcloud", "storage", "cp", "*nsys-rep", self.output_path])
            else:
                print(
                    "WARNING: gcloud is not installed or not found in the system's PATH. Skipping upload...")
        elif self.mode == "xplane":
            jax.profiler.stop_trace()
        else: 
            raise ValueError("Profiler mode is not supported. Supported modes are xplane and nsys.")
        
        self._is_tracing = False
        print(f"Profiler completed tracing. Profile saved to {self.output_path}")