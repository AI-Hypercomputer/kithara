.. _scaling_with_ray:

Running on Multihost?
=====================

Kithara works with any accelerator orchestrator. However, if you are new to distributed training, we provide guide for multihost training with `Ray <https://docs.ray.io/en/latest/index.html>`_.

.. admonition:: What is Ray?

    Ray is a popular and powerful tool for running distributed TPU and GPU workloads, offering:

    - Dashboard for job queueing and log visualization
    - Streamlined environment setup and file syncing
    - Simple command interface for multihost workloads

Follow the instructions below to set up your Ray cluster.

* :doc:`Set up Ray Cluster with TPU VMs <installation/tpu_vm>`
* :doc:`Set up Ray Cluster with TPU QRs <installation/tpu_qr>` 
* :doc:`Set up Ray Cluster with TPU GKE <installation/tpu_gke>`

.. toctree::
    :maxdepth: 2
    :caption: Ray Cluster Setup
    :hidden:

    Ray Cluster with TPU VMs <installation/tpu_vm>
    Ray Cluster with TPU QRs <installation/tpu_qr>
    Ray Cluster with TPU GKE <installation/tpu_gke>


After setting up your Ray Cluster, you can scale up your workload to run on multiple hosts using the following recipe.

.. code-block:: python
    :caption: my_multihost_ray_job.py
    
    import ray
    import jax

    ray.init()

    num_chips_per_host = 4  # <--IMPORTANT: Use 4 for v4 and v5, 8 for v4e and v5e TPU
    num_tpu_hosts = int(ray.cluster_resources()["TPU"] / num_chips_per_host)
    print(f"{num_tpu_hosts=}")

    @ray.remote(resources={"TPU": num_chips_per_host})
    def main():

        # HuggingFace login
        from huggingface_hub import login
        import os

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        # Let JAX know that we are running a distributed job
        jax.distributed.initialize()

        # No need to change your single host job script, simply use it as it is. 
        from examples.singlehost.quick_start import run_workload

        # Run this workload on all hosts. Don't worry, we are handling 
        # model sharding and batch sharding for you. 
        run_workload()


    ray.get([main.remote() for i in range(num_tpu_hosts)])

    ray.shutdown()

To launch this job, run the following command::

    ray job submit \
    --address="http://localhost:8265" \
    --runtime-env-json='{"env_vars": {"HF_TOKEN": "your_token_here", "HF_HUB_ENABLE_HF_TRANSFER": "1"}}' \
    -- "python3.11 my_multihost_ray_job.py"

Equivalently, use the Kithara `helper script <https://github.com/AI-Hypercomputer/kithara/blob/main/ray/submit_job.py>`_::

    python ray/submit_job.py "python3.11 my_multihost_ray_job.py" --hf-token your_token

Check out some multihost examples: 

- `Multihost Quickstart Example <https://github.com/AI-Hypercomputer/kithara/blob/main/examples/multihost/ray/TPU/quick_start.py>`_
- `Multihost Continued Pretraining Example <https://github.com/AI-Hypercomputer/kithara/blob/main/examples/multihost/ray/TPU/continued_pretraining_example.py>`_
- `Multihost SFT+LoRA Example <https://github.com/AI-Hypercomputer/kithara/blob/main/examples/multihost/ray/TPU/sft_lora_example.py>`_
