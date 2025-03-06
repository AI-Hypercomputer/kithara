.. _vllm:

Serve with vLLM
==============

The fine-tuned model can be served on `vLLM:GPU <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html>`_ or `vLLM:TPU <https://docs.vllm.ai/en/latest/getting_started/tpu-installation.html>`_.

Exporting your model
--------------------
In order to serve your finetuned model, first make sure that your output checkpoints are exported to a GCS bucket or persistent volume. You can do this byoverriding the `model_output_dir` parameter in the config yaml:

.. code-block:: python

    model_output_dir: "gs://bucket_name/ckpt/"


Starting vLLM with a Finetuned Model
------------------------------------
Next, start your vLLM deployment by attaching the previous output volume as a mounted drive. Please visit our GCS guide (link TBD) for details.

You can start vLLM with a path to the directory containing the model files. For example if you mounted the volume at `/model` and the model checkpoints are located under the `checkpoint` directory:

.. code-block:: bash 

    vllm serve /model/checkpoint --tensor_parallel_size 8 --max-model-length 4096

where `tensor_parallel_size` is equal to the number of TPU chips available. You can adjust `max-model-length` depending on the availably HBM.

If loading the model is successful, you should see the following output in the console:

.. code-block:: bash

    INFO 03-05 23:24:13 api_server.py:756] Using supplied chat template:
    INFO 03-05 23:24:13 api_server.py:756] None
    INFO 03-05 23:24:13 launcher.py:21] Available routes are:
    INFO 03-05 23:24:13 launcher.py:29] Route: /openapi.json, Methods: GET, HEAD
    INFO 03-05 23:24:13 launcher.py:29] Route: /docs, Methods: GET, HEAD
    INFO 03-05 23:24:13 launcher.py:29] Route: /docs/oauth2-redirect, Methods: GET, HEAD
    INFO 03-05 23:24:13 launcher.py:29] Route: /redoc, Methods: GET, HEAD
    INFO 03-05 23:24:13 launcher.py:29] Route: /health, Methods: GET
    INFO 03-05 23:24:13 launcher.py:29] Route: /ping, Methods: POST, GET
    INFO 03-05 23:24:13 launcher.py:29] Route: /tokenize, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /detokenize, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /v1/models, Methods: GET
    INFO 03-05 23:24:13 launcher.py:29] Route: /version, Methods: GET
    INFO 03-05 23:24:13 launcher.py:29] Route: /v1/chat/completions, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /v1/completions, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /v1/embeddings, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /pooling, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /score, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /v1/score, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /rerank, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /v1/rerank, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /v2/rerank, Methods: POST
    INFO 03-05 23:24:13 launcher.py:29] Route: /invocations, Methods: POST
    INFO:     Started server process [67252]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


Send a request to the vLLM server to verify that it's working:

.. code-block:: bash

    curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "/model/checkpoint/",
           "prompt": "your prompt",
           "max_tokens": 256,
           "temperature": 0.7
        }'

