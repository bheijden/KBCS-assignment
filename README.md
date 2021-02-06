# Vision-based angle prediction (SC42050 - Knowledge Based Control Systems)

## Installation instructions
To set up the code locally, either download the ZIP file from Brightspace, or clone the Github repository to your personal computer. 

You can clone (tested on Linux) the repository to your current working directory with:
```
git clone https://github.com/bheijden/KBCS-assignment.git
```
After you extracted all the files somewhere locally from the ZIP file, or after you cloned the Github repository, move into that directory. 
In a terminal on either Linux/MAC OS, you can do so with (change ```INSERT_LOCAL_PATH_TO``` accordingly):
```
cd INSERT_LOCAL_PATH_TO/KBCS-assignment
```
*Note: it could be that the directory is called* `KBCS-assignment-main` *instead of* `KBCS-assignment`.

To run the code, we require the following dependencies. Install them so that your python 3.7 interpreter can access them. 
You can do this manually, or by creating an anaconda virtual environment (see next instruction).
```
- matplotlib
- psutil
- tensorflow==2.3.1
- gym
- colorama
- termcolor
- scikit-image
```
The simplest way to install all required dependencies is to create an anaconda environment and add the dependencies there.
Links for installing anaconda are provided at the end of this instruction.

With anaconda installed, you can manually install the aforementioned list of dependencies to your environment. 
Or, directly create a new environment (tested on Linux) with all required dependencies by running:
```
conda env create -f conda_env.yml
```
After the installation ends, activate the environment ```kbcs``` with:
```
conda activate kbcs
```
Once inside the virtual environment, (*Note: you are inside the virtual environment if you see (kbcs) in front of the terminal line*), you can run the python scripts with:
```
python generate_data.py
python train.py
```
Note that `train.py` will only run after you make appropriate changes to the code as specified in the assignment.

You can find information about installing anaconda here:

Ubuntu 18.04: https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

Ubuntu 20.04: https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/

Windows: https://docs.anaconda.com/anaconda/install/windows/

Mac OS: https://docs.anaconda.com/anaconda/install/mac-os/


