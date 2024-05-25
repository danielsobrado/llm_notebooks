# Setting Up Mamba, Installing `unsloth` in WSL2, and Adding Conda Environment as a Kernel in Jupyter

Install the `unsloth` package, and make your Conda environment available as a kernel in Jupyter, on a system running WSL2 on Windows 11 with an Nvidia 40XX GPU.

## Step 1: Install Mamba

First, download and install Mambaforge:

```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh
```

## Step 2: Create and Activate a Conda Environment

Create a new Conda environment named `unsloth_env` with Python 3.10:

```
mamba create --name unsloth_env python=3.10
mamba activate unsloth_env
```

## Step 3: Install Required Packages

Install the necessary packages including `pytorch`, `cudatoolkit`, `xformers`, and `bitsandbytes`:

```
mamba install cudatoolkit xformers bitsandbytes pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -c xformers -c conda-forge
```

## Step 4: Upgrade Pip

Upgrade `pip` to the latest version to avoid installation issues:

```
pip install --upgrade pip
```

## Step 5: Install the `unsloth` Package

Install the `unsloth` package from the GitHub repository:

```
pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"
```

## Step 6: Install and Configure Jupyter Kernel

Install `ipykernel` and add the Conda environment as a kernel in Jupyter:

```
mamba install ipykernel
python -m ipykernel install --user --name=unsloth_env --display-name "Python (unsloth_env)"
```

## Step 7: Verify Kernel Installation

Check if the kernel is successfully added to Jupyter:

```
jupyter kernelspec list
```

## Step 8: Start Jupyter Notebook or JupyterLab

Start Jupyter Notebook or JupyterLab to use the newly added kernel:

```
jupyter notebook
```

or

```
jupyter lab
```

## Troubleshooting

### ImportError: cannot import name 'packaging' from 'pkg_resources'

If you encounter the following error:

```
ImportError: cannot import name 'packaging' from 'pkg_resources' (/home/drusniel/mambaforge/envs/unsloth_env/lib/python3.10/site-packages/pkg_resources/__init__.py)
```

You can resolve it by installing a specific version of `setuptools` using Mamba:

```
mamba install setuptools==62.6.0
```

## Environment Details

- **Operating System:** Windows 11 with WSL2
- **GPU:** Nvidia 40XX

By following these steps, I was able to set up my environment, install the necessary packages, and configure the Jupyter kernel correctly.
