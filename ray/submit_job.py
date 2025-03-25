from ray.job_submission import JobSubmissionClient, JobStatus
import time
import argparse


def wait_until_status(client, job_id, status_to_wait_for, timeout_seconds=60):
    start = time.time()
    while time.time() - start <= timeout_seconds:
        status = client.get_job_status(job_id)
        print(f"status: {status}")
        if status in status_to_wait_for:
            break
        time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit a Ray job with specified entrypoint"
    )
    parser.add_argument(
        "job_file",
        type=str,
        help='The entrypoint script to run (e.g., "python examples/multihost/ray/TPU/sft_lora_example.py")',
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="http://localhost:8265",
        help="Ray cluster address (default: http://localhost:8265)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional. Your Huggingface token if HuggingFace login is need. ",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    client = JobSubmissionClient(args.ray_address)
    print("Initialized job client")

    job_id = client.submit_job(
    entrypoint=args.job_file,
        runtime_env={
            "working_dir": "./",
            # These files are too large and are safe to exclude
            "excludes": ["**/test_assets/", ".git/", "*.Dockerfile", "*.pkl"],
            "pip": [],  # Any missing dependencies to run job goes here
            "env_vars": {"HF_TOKEN": args.hf_token, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        },
    )

    print(f"Submitted job with {job_id=}")

    wait_until_status(
        client, job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}
    )
    
    # See full logs in the Ray Dashboard
    logs = client.get_job_logs(job_id)
    print(logs)

if __name__ == "__main__":
    
    main()
    
    