# Neural network assignment (SC42050 - Knowledge Based Control Systems)

## Installation instructions
With your terminal, move into the directory
```
cd assignment_KBC
```
To run the code, we require the following dependencies. Install them so that your python interpreter can access them.
```
- python=3.7
- matplotlib
- psutil
- tensorflow==2.3.1
- gym
- colorama
- termcolor
- scikit-image
```
The simplest way to install all required dependencies is to create an anaconda environment and add the dependencies there. 
With anaconda installed, you can directly create a new environment with all dependencies by running:
```
conda env create -f conda_env.yml
```
After the installation ends, activate the environment ```kbcs``` with:
```
conda activate kbcs
```
With the environment activated, you can run the python scripts with:
```
python generate_data.py
python train.py
```
You can find information about installing anaconda here:

Ubuntu 18.04: https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

Windows: INSERT LINK

Mac OS: INSERT LINK

