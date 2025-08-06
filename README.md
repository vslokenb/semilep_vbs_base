# Installation Pocket Coffea in your system lxplus

The simplest and most recommended way to use PocketCoffea in an environment like CERN lxplus is through an Apptainer image. This method is crucial to ensure a stable working environment with all dependencies pre-installed, especially if you plan to scale your analysis on the HTCondor system.

An Apptainer image is a virtual environment that isolates the code and contains everything needed for PocketCoffea to work properly (the appropriate version of Python and all dependencies). This allows you to work without having to install anything manually.

## How does the apptainer image work?

The images are built and managed in several stages:

1. Docker images: The process starts with a Docker image containing the Python environment with PocketCoffea installed.

2. Versions: There are different versions so you can choose the one that best suits your need:

  * main: Contains the most recent code (*-latest). It is ideal for developers who want to test the latest features.

  
  * stable: This is the recommended and most tested version (*-stable). It offers the best stability for most users.

  * tagged: These are specific, numbered versions of the code.

3. Conversion and distribution: Docker images are “unpacked” and converted to Apptainer images. These are stored on a shared file system called CVMFS, making them easily accessible to all users in analysis environments such as LXPLUS.

## First steps in LXPLUS

To start working with PocketCoffea in lxplus, you only need to execute one command to access the Apptainer environment.

Step 1: Run the following command in your lxplus terminal to enter an interactive environment (singularity) with PocketCoffea.

<br>

```
apptainer shell -B /afs -B /cvmfs/cms.cern.ch \
                -B /tmp  -B /eos/cms/  -B /etc/sysconfig/ngbauth-submit \
                -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" \
    /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable

```

<br>

Step 2: Choose the version

The last part of the command (...pocketcoffea:lxplus-el9-stable) allows you to choose the version of the image you want to use. Although the stable version is the most recommended, you can change it to latest if you need the latest updates.

Once you are inside the environment, no installation is necessary. PocketCoffea scripts are available globally, so you can start running your analysis now.


Step 3:  Check the installation

Once you have accessed the Apptainer environment, you can confirm that PocketCoffea is ready to use. Simply run the following command in the terminal:

<br>

```
pocket-coffea

```

<br>

If everything is configured correctly, you will see an output similar to this:

<br>

```

Singularity> pocket-coffea 

    ____             __        __  ______      ________
   / __ \____  _____/ /_____  / /_/ ____/___  / __/ __/__  ____ _
  / /_/ / __ \/ ___/ //_/ _ \/ __/ /   / __ \/ /_/ /_/ _ \/ __ `/
 / ____/ /_/ / /__/ ,< /  __/ /_/ /___/ /_/ / __/ __/  __/ /_/ /
/_/    \____/\___/_/|_|\___/\__/\____/\____/_/ /_/  \___/\__,_/


Running PocketCoffea version 0.9.9
- Documentation page:  https://pocketcoffea.readthedocs.io/
- Repository:          https://github.com/PocketCoffea/PocketCoffea

Run with --help option for the list of available commands 

```

<br>

This output with the logo and software version confirms that the PocketCoffea environment is active and ready to be used. Now you can start running your analysis.


# Manual installation in a Python environment

If you only need to run PocketCoffea on your local machine and not on batch processing systems like HTCondor, you can install the package directly in a Python environment. You have several options for this, such as conda or virtualenv.

Step 1: Clone the repository

Clone the PocketCoffea repository from GitHub to the location of your choice.

<br>

```
git clone git@github.com:PocketCoffea/PocketCoffea.git
cd PocketCoffea

```

<br>


Step 2: Create and activate a Python environment

Now, you must set up a Python environment to isolate the package dependencies. Choose one of the following options:

Option A: Using LCG and venv.

This option is useful if you want to use a base LCG environment to avoid downloading many packages.

1. Create the environment (first time only):

<br>

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_103/x86_64-centos7-gcc11-opt/setup.sh
python -m venv --system-site-packages myenv


# python3 -m venv --system-site-packages myenv

```

<br>

2. Activates the environment:

<br>

```
source myenv/bin/activate

```

<br>
   
3. Install PocketCoffea and Dependencies

With your new virtual environment active, use pip to install PocketCoffea in editable mode. This command will also download and install all of the required dependencies (such as coffea, numba, and rucio-clients) directly into your myenv environment.

<br>

```
pip install -e .

```

<br>

4. Verify the Installation

After the installation is complete, you can test it by running the pocket-coffea command. This should display the PocketCoffea welcome message and version number.

<br>

```
# Run PocketCoffea from your environment
pocket-coffea

```

<br>

If the command runs successfully, you'll see an output similar to this, confirming that PocketCoffea is installed and ready to use:

<br>

```

Singularity> pocket-coffea 

    ____             __        __  ______      ________
   / __ \____  _____/ /_____  / /_/ ____/___  / __/ __/__  ____ _
  / /_/ / __ \/ ___/ //_/ _ \/ __/ /   / __ \/ /_/ /_/ _ \/ __ `/
 / ____/ /_/ / /__/ ,< /  __/ /_/ /___/ /_/ / __/ __/  __/ /_/ /
/_/    \____/\___/_/|_|\___/\__/\____/\____/_/ /_/  \___/\__,_/


Running PocketCoffea version 0.9.9
- Documentation page:  https://pocketcoffea.readthedocs.io/
- Repository:          https://github.com/PocketCoffea/PocketCoffea

Run with --help option for the list of available commands 

```

<br>
