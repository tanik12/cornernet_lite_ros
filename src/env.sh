#!/bin/bash

echo "プログラム開始"

export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

cd ~/catkin_ws_python3
pyenv shell anaconda3-2019.10
source activate CornerNet_Lite
source ~/catkin_ws_python3/devel/setup.bash
rosrun cornernet_lite_ros demo_cam_Squeeze.py
#python cornernet_lite_ros demo_cam_Squeeze.py
echo "プログラム終了"
