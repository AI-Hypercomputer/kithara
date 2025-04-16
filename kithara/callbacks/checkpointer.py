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



from keras.src.callbacks.callback import Callback
import jax
import orbax.checkpoint as ocp
from typing import Optional
import os
from kithara.optimizers.protocol import KitharaOptimizer

CheckpointManager = ocp.CheckpointManager
CheckpointManagerOptions = ocp.CheckpointManagerOptions

class Checkpointer(Callback):
    """A callback for saving and loading model checkpoints during 
        training.

    This class provides functionality to automatically save model 
    checkpoints at specified intervals during training, and to load 
    checkpoints for model restoration. It can be used either as a callback
    during training or as a standalone checkpointing utility.

    Args:
        checkpoint_dir (str): Directory path where checkpoints will be saved.
            Can be a GCS path or a local directory
        model (Optional[kithara.Model]): The model instance to checkpoint. 
            Required for standalone use.
        use_async (bool): Whether to use asynchronous checkpointing. Defaults 
            to True.
        save_interval_steps (int): Number of steps between checkpoints. 
            Use -1 to disable automatic saving. Defaults to 100.
        max_to_keep (int): Maximum number of checkpoints to keep. Older checkpoints 
            are deleted.Defaults to 5.
        by_batch (bool): Whether to save checkpoints based on batch count. 
            Defaults to True.
        by_epoch (bool): Whether to save checkpoints based on epoch count. 
            Defaults to False.
        optimizer (Optional[KitharaOptimizer]): The optimizer instance to checkpoint.

    Example:
        ```
        # Use as a training callback
        checkpointer = Checkpointer("gs://...", save_interval_steps=100, max_to_keep=5)
        trainer = Trainer(..., checkpointer=checkpointer)
        trainer.train()

        # Use as a standalone utility
        model = kithara.MaxTextModel.from_preset("hf://google/gemma2-2b")
        checkpointer = Checkpointer("gs://...", model)
        
        # Save checkpoint
        checkpointer.save(0, blocking=True)
        
        # Load latest checkpoint back into model
        checkpointer.load()
        ```
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model: Optional['kithara.Model'] = None,
        use_async: bool = True,
        save_interval_steps: int = 100,
        max_to_keep:int = 5,
        by_batch: bool = True, 
        by_epoch: bool = False, 
        optimizer: Optional['KitharaOptimizer'] = None,
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.use_async = use_async
        self.save_interval_steps = save_interval_steps
        self.max_to_keep = max_to_keep
        self.by_batch = by_batch
        self.by_epoch = by_epoch
        assert (self.by_batch and not self.by_epoch) or (self.by_epoch and not self.by_batch), "One and only one of `by_batch` and `by_epoch` should be True."
        
        self.mngr = self._set_up_checkpoint_manager()
        self._num_train_batch = 0
        self._num_train_epoch = 0
        if model: 
            self._model = model
        assert self._model is not None, "Please provide the model instance when creating the Checkpointer."
        if optimizer:
            self._optimizer = optimizer
        else:
            self._optimizer = None

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of every training batch."""
        self._num_train_batch += 1
        
        if self.save_interval_steps>0 and self.by_batch:
            if self._num_train_batch % self.save_interval_steps == 0:
                self.save(self._num_train_batch)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of every training epoch."""
        self._num_train_epoch += 1
        
        if self.save_interval_steps>0 and self.by_epoch:
            if self._num_train_epoch % self.save_interval_steps == 0:
                self.save(self._num_train_epoch)

    def on_train_end(self, logs=None):
        # Block since checkpointing is could be async
        self.mngr.wait_until_finished()
        
    def save(self, step, blocking = False):
        """Saves a checkpoint of the model's current state.

        Args:
            step (int): The current step number (used in checkpoint filename)
            blocking (bool): If True, waits for the checkpoint to be 
                fully written before returning. Defaults to False.

        Note:
            If a preemption step is reached, this method will force blocking behavior,
            save the checkpoint, and exit the program to ensure state is preserved.
        """
        print(f"-> Saving checkpoint after {step} training steps/epochs...")
        state = {
            v.path: v.value for v in self.model.variables
        }
        if self._optimizer:
            state = state | {
            v.path: v.value for v in self._optimizer.variables
        }
        jax.block_until_ready(state)
        
        self.mngr.save(step, args=ocp.args.StandardSave(state))

        # If we are at a preemption step, we must wait for the 
        # checkpoint to finish writing before exiting.
        if self.mngr.reached_preemption(step):
            print("-> Being preempted, saving checkpoint before exiting...")
            self.mngr.wait_until_finished()
            print(f"✅ Successfully saved checkpoint to {os.path.join(self.checkpoint_dir, str(step))}")
            exit()
        
        if blocking: 
            self.mngr.wait_until_finished()
            print(f"✅ Successfully saved checkpoint to {os.path.join(self.checkpoint_dir, str(step))}")
        
    def load(self, step=None, in_place=True):
        """Loads a checkpoint into the model.

        Args:
            step (Optional[int]): The specific checkpoint step to load. If None, loads the latest
                checkpoint. Defaults to None.
            in_place (bool): If True, updates the model's variables with the loaded state.
                If False, only returns the state without updating the model. Defaults to True.

        Returns:
            dict: The loaded checkpoint state, mapping variable paths to their values.
        """

        if step is None:
            step = self.mngr.latest_step()
        
        state = {
            v.path: v.value for v in self.model.variables
        }
        if self._optimizer:
            state = state | {
                v.path: v.value for v in self._optimizer.variables
            }
        abstract_state = jax.tree.map(ocp.tree.to_shape_dtype_struct, state)

        def set_dtype(abstract_arr):
            return abstract_arr

        state = self.mngr.restore(step, args=ocp.args.StandardRestore(
            jax.tree.map(set_dtype, abstract_state)))
        
        if in_place: 
            for v in self.model.variables:
                new_var = state[v.path]
                v.assign(new_var)
            for v in self._optimizer.variables:
                new_var = state[v.path]
                v.assign(new_var)

        return state
        
    def _set_up_checkpoint_manager(self) -> 'CheckpointManager':

        # First convert checkpoint_dir to absolute path
        if self.checkpoint_dir.startswith("gs://") is False:
            self.checkpoint_dir = os.path.abspath(self.checkpoint_dir)
        
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.save_interval_steps,
            max_to_keep=self.max_to_keep,
            enable_async_checkpointing=self.use_async
        )
        mngr = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=options,
        )
        return mngr
