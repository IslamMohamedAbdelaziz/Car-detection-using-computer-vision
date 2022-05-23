#!/bin/bash
python3 Lane_line_detection.py $1 $2 $3

#create virtual enviroment
python3 -m venv env

#activate virtual enviroment
source env/bin/activate

#install dependencies
python3 -m pip install pillow
python3 -m pip install matplotlib 
python3 -m pip install scikit-build 
python3 -m pip install joblib 
python3 -m pip install opencv-python 
python3 -m pip install imutils 
python3 -m pip install numpy 
python3 -m pip install imageio
python3 -m pip install moviepy 
python3 -m pip install ez_setup 
python3 -m pip install IPython

# pathing input parameters
python3 Lane_line_detection.py $1 $2 $3

