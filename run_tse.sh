if [ "$#" -lt 1 ]
then 
    CUDA_VISIBLE_DEVICES=-1 python /SSD4TB/skygun/robot_code/PyLapras/agent/StrictTSEAgent.py
else
    CUDA_VISIBLE_DEVICES="$1" python /SSD4TB/skygun/robot_code/PyLapras/agent/StrictTSEAgent.py
fi