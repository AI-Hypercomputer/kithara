# Orchestration

You can find instructions for running multihost workloads in this section.

## Enable Multihost Tuning via Ray Cluster

Ray is a great tool for running distributed TPU and GPU workloads. It offers a dashboard for job queueing and log visualization, and streamlines the environment set up and file syncing. Once you set up a Ray Cluster, you can run multihost workloads with a simple command. We provide instructions for setting up Ray Cluster with GCE TPU/GPU VMs and TPU QRs.

### Instructions for setting up Ray Cluster with GCE resources

1. Assume you have resource capacity and quota in your GCP project and region/zone. Modify `ray/TPU/GCE/cluster.yaml` or `ray/GPU/cluster.yaml`template with your configurations. Please take a look at the YAML file and ctrl+F for MODIFY.

2. Run the following commands to bring up your ray cluster.

   For TPU,

   ```
   cd ray/TPU/GCE
   ```

   For GPU:

   ```
   cd ray/GPU
   ```

   Run

   ```
   ray up -y cluster.yaml
   ```

   Monitor the node set up process by running

   ```
   ray monitor cluster.yaml
   ```

   While you monitor the node set up process, launch the ray dashboard.

   ```
   ray dashboard cluster.yaml
   ```

   You should see the dashboard on your `localhost:8265`

   **Troubleshooting**:

   - If you see an `update-failed` error, don't worry. This should not affect the nodes being properly set up.
   - You can exec into the ray head node using `ray attach cluster.yaml` and run `ray status` to see the status of the nodes.

3. Once all nodes in your ray cluster are set up and active, and you have launched the dashboard, you are ready to run multihost jobs.

   First either clone the Kithara Github Repo or copy the necessary scripts to your local machine.
   Then, use the `ray/submit_job.py` helper script to run the workload on your Ray Cluster.

   ```
   python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/sft_lora_example.py" --hf-token your_token
   ```

   You can early-stop your job using

   ```
   export RAY_ADDRESS="http://127.0.0.1:8265"
   ray job stop ray_job_id
   ```

4. Once you are done with your ray cluster, tear it down

   `ray down cluster.yaml`

### Instructions for setting up Ray Cluster with TPU QR

1. Create the Ray head node and launch the Ray cluster.
   ```
   ray up -y ray/TPU/QR/cluster.yaml
   ```
2. The `ray up` command will take a few minutes, if for any reason you lost the terminal,
   you can use the following command to monitor the cluster creation process.
   ```
   ray monitor ray/TPU/QR/cluster.yaml
   ```
3. Launch Ray Cluster dashboard once the `ray_head_default` node is `Active`. You should see the dashboard on your `localhost:8265`

   ```
   ray dashboard ray/TPU/QR/cluster.yaml
   ```

4. In the Ray dashboard, note down the IP of your ray head node, you can find this in the `Cluster` panel.

   _Note that this should be the internal ip not the external ip of the head node. It should start with 10.x.x.x_

5. Specify the following variables with your own TPU resources spec.

   ```
    export ZONE="us-central2-b"
    export QR_NAME="my_tpu_qr"
    export NODE_ID="v4-a"
    export PROJECT="gcp_project_name"
    export TPU_TYPE="v4-32"
    export RAY_CLUSTER_IP="your_cluster_ip"
   ```

6. Create TPU VMs via QR.

   ```
   gcloud alpha compute tpus queued-resources create $QR_NAME --node-id $NODE_ID --zone $ZONE  --project $PROJECT --accelerator-type $TPU_TYPE --runtime-version tpu-ubuntu2204-base --metadata-from-file='startup-script=ray/TPU/QR/qr_worker_startup_script.sh' 
   ```

   If you are using DWS, use the following command: 

   ```
   gcloud alpha compute tpus queued-resources create $QR_NAME --node-id $NODE_ID --zone $ZONE  --project $PROJECT --accelerator-type $TPU_TYPE --runtime-version tpu-ubuntu2204-base --metadata="queued_resource_tier_b=true" --metadata-from-file='startup-script=ray/TPU/QR/qr_worker_startup_script.sh' 
   ```

7. Monitor the status of the QR creation with the following command.

   ```
   gcloud compute tpus queued-resources describe $QR_NAME --project $PROJECT --zone $ZONE
   ```
   Once the status of the QR becomes `ACTIVE`, monitor the logs to make sure that the packages have finished installation. 

   ```
   gcloud alpha compute tpus queued-resources ssh $QR_NAME --project $PROJECT --zone $ZONE --command="sudo cat /var/log/syslog | grep startup-script" --worker=0 --node=all
   ```

8. Once the QRs are ready, attach the TPU VMs to the Ray Cluster as worker nodes.

   ```
   gcloud alpha compute tpus queued-resources ssh $QR_NAME --project $PROJECT --zone $ZONE --command="ray start --address=$RAY_CLUSTER_IP:6379 --resources='{\"tpu_host\": 1}'" --worker=all --node=all
   ```

   **Troubleshooting:** If you run into Python or Ray version inconsistencies, check the logs of the worker nodes, and see if anything went wrong in the installation process. 

   ```
   gcloud alpha compute tpus queued-resources ssh $QR_NAME --project $PROJECT --zone $ZONE --command="sudo cat /var/log/syslog | grep startup-script" --worker=all --node=all
   ```

   gcloud alpha compute tpus queued-resources ssh $QR_NAME --project $PROJECT --zone $ZONE --command="sudo lsof /dev/accel*" --worker=0 --node=all


9. Now your Ray Cluster is ready, try out examples in the `examples/multihost/TPU` folder.

   First either clone the Kithara Github Repo or copy the necessary scripts to your local machine.
   Then, use the `ray/submit_job.py` helper script to run the workload on your Ray Cluster. 

   ```
   python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/sft_lora_example.py" --hf-token your_token
   ```
   
   You can early-stop your job using

   `ray job stop ray_job_id`

10. To remove QRs from your Ray Cluster, run this command.

    ```
    gcloud compute tpus queued-resources delete $QR_NAME --project $PROJECT --zone $ZONE
    ```

11. Once you are done with your ray cluster, tear it down

    `ray down ray/TPU/QR/cluster.yaml`

## TroubleShooting

1. Error `Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by process with pid <PID>.`

   Try removing the following files from TPU VM hosts by running these two lines at the beginning of the Ray remote function.

   ```
   import subprocess
   subprocess.run(["rm", "-rf", "/tmp/libtpu_lockfile", "/tmp/tpu_logs"])
   ```
