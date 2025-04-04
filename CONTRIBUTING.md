# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.


# Set up instructions for contributors

## Developing on single host

### 1. Clone this repo on your TPU or GPU VM

```
git clone --recursive https://github.com/AI-Hypercomputer/kithara.git
```

Make sure that you have cloned Kithara with the MaxText and JetStream submodule. If already cloned, add the submodule using the following command.

```
git submodule update --init --recursive
```

**Troubleshooting**:

If you don't see the MaxText or Jetstream submodule after cloning or updating, try

```
git submodule add --force https://github.com/google/maxtext kithara/model/maxtext/maxtext
git submodule add --force https://github.com/AI-Hypercomputer/JetStream kithara/model/maxtext/JetStream
```

### 2. Install dependencies

Kithara requires `Python>=3.11`. First create a virtual environment with this Python version.

1. With `conda`:
   ```
   conda create -n kithara_env python=3.11
   conda activate kithara_env
   ```
2. With `venv`:

   ```
   python3.11 -m venv kithara_env
   source kithara_env/bin/activate
   ```

Then, pip install the Kithara library in editable mode.

```
# On TPU
pip install -e .[tpu,dev] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --extra-index-url https://download.pytorch.org/whl/cpu


# On GPU
pip install -e .[gpu,dev]

# On CPU
pip install -e .[cpu]
```

Now you should be ready to develop on single host! Test your set up by running a script in the `examples/singlehost` folder.

### Developing on MultiHost

It is important to set up development on MultiHost for implementing multihost features, and to thoroughly test that your single host code work on all multihost environments. Follow the steps below to set up multi-host development with Ray.

1. Clone the repo on your local machine (i.e. laptop, CloudTop, or VM) and install the dependencies. 

   ```
   git clone --recursive https://github.com/AI-Hypercomputer/kithara.git
   # Use conda or venv 
   conda create -n kithara_env python=3.11
   conda activate kithara_env
   # Instead Kithara in edit mode
   pip install -e .[cpu]
   ```

2. Set up your Ray Cluster

   Follow the instructions in `ray/README.md` to set up your Ray Cluster. It is preferable to use the `GCE` option instead of `QR`.

3. By default, Ray uses the pip-installed version of Kithara specified in the cluster configuration YAML. To use your local development version instead, simply add the following code to your Ray remote function:

   ```
   @ray.remote(resources={"TPU": num_chips_per_host})
   def main():
       import subprocess
       subprocess.run(["pip install -e .[tpu] --no-deps"], shell=True)
       # Your code follows
   ```

   The --no-deps flag prevents reinstalling dependencies.

   Now you should be ready to develop on multihost! Please test your set up by running an example in the `examples/multihost` folder. E.g.

   ```
   python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/sft_lora_example.py" --hf-token your_token
   ```

# Releasing a New Version of Kithara (Googlers Only)

These instructions are for Googlers with access to Kithara's credentials via go/valentine. Search for "kithara" on go/valentine to retrieve the necessary credentials for TestPyPI and PyPI.

### Release Steps

1. **Run Tests**: Execute all tests and confirm they pass successfully. This will take ~10 minutes. :

   ```
   python -m unittest
   ```

2. **Update Version Number**: Modify the `pyproject.toml` file, incrementing the version number according to the MAJOR.MINOR.PATCH convention. For example, if releasing version 1.2.3, and the next release is a patch, change it to 1.2.4.

3. **Build the Wheel File**: Create the distribution wheel file. After running the following command, you should see a `.whl` file and a `.tar.gz` file created under `/dist`.

   ```
   flit build --no-use-vcs
   ```

4. **Upload to TestPyPI**.

   - Configure .pypirc: Create (or update) the `$HOME/.pypirc` file with the TestPyPI credentials found in go/valentine. This file tells twine where and how to upload.

   - Upload the newly built wheel file to TestPyPI:

     ```
     twine upload --repository testpypi dist/*
     ```

5. **Test on a Fresh TPU VM**:

   - **Set up Conda**: If Conda isn't already installed, follow these steps:

     ```
     mkdir -p ~/miniconda3
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
     bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
     rm ~/miniconda3/miniconda.sh
     ```

     After installing Conda, close and reopen your terminal application or refresh it by running the following command:

     ```
     source ~/miniconda3/bin/activate
     ```

     To initialize conda on all available shells, run the following command:

     ```
     conda init --all
     ```

   - **Create and Activate Conda Environment**:

     ```
     conda create -n test python==3.11
     conda activate test
     ```

   - **Install Kithara from TestPyPI**: This command installs Kithara and its dependencies. It prioritizes TestPyPI for finding packages but will fall back to PyPI. Please replace VERSION with the release version.

     ```
     pip install kithara[tpu]==VERSION --index-url https://pypi.org/simple --extra-index-url https://test.pypi.org/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --extra-index-url  https://download.pytorch.org/whl/cpu --force-reinstall
     ```

   - **Resolve Dependency Conflicts (Important)**: Due to the mixed PyPI/TestPyPI installation, you might encounter dependency conflicts. Currently, the following packages need to be specifically reinstalled from PyPI:

     ```
     pip uninstall ml-goodput-measurement -y && pip install ml-goodput-measurement --index-url https://pypi.org/simple

     pip uninstall chex -y && pip install chex>=0.1.88 --index-url https://pypi.org/simple
     ```

     _Note: This list may change. If you encounter other issues, compare package versions with `pip show pacakge_name` between this new environment and your own development environment and reinstall any conflicting packages from PyPI._

6. **Run Single-Host End-to-End Examples**: Clone the Kithara repository (using the correct branch) and execute the tests to make sure everything works as expected.

   ```
   git clone -b BRANCH_NAME https://github.com/AI-Hypercomputer/kithara.git
   ```

   Remove the
   `kithara` source code folder just in case.

   ```
   cd kithara
   rm -rf kithara
   ```

   Log into HuggingFace with your own API token

   ```
   huggingface-cli login
   ```

   Run the set of light tests.

   ```
   RUN_LIGHT_TESTS_ONLY=1 python -m unittest
   ```

   Run the e2e tests.

   ```
   python -m unittest tests/trainer/test_sft_e2e.py
   ```

7. **Run Multi-Host End-to-End Examples**: Spin up a new Ray Cluster, updating the `cluster.yaml` file with the TestPyPI `kithara` dependency, and test running a multihost script.

   ```
   python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/sft_lora_example.py" --hf-token your_token
   ```

8. **Upload to PyPI**: Once testing on TestPyPI is successful, you'll need to upload the wheel to the official PyPI repository. Don't forget to rerun `flit build --no-use-vcs` if you have made changes and need to rebuild the wheel. 

   ```
   twine upload --repository pypi dist/*
   ```

9. **Create Release Branch**: Merge your changes into main and create a release branch.  

    First merge your changes into the main branch. 
    
    Next, create a release branch. 
    ```
    git checkout -b release/v0.0.5
    git push -u origin release/v0.0.5
    ```

    After creating the release branch, go to `Releases` on GitHub, click `Create new release`. 

        Tag: v0.0.5
        Target: release/v0.0.5
        Title: Kithara v0.0.5
        Add release notes
        Click "Publish release"


    After release, merge release branch into main. 
    ```
    git checkout main
    git merge release/v0.0.5
    git push
    ```
