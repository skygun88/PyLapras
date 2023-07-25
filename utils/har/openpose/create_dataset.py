import os
import sys
import numpy as np
import pandas as pd

HEADER = 'nose_x,nose_y,neck_x,neck_y,Rshoulder_x,Rshoulder_y,Relbow_x ,Relbow_y,Rwrist_x,RWrist_y ,LShoulder_x,LShoulder_y,LElbow_x,LElbow_y,LWrist_x,LWrist_y ,RHip_x,RHip_y,RKnee_x,RKnee_y,RAnkle_x,RAnkle_y,LHip_x,LHip_y,LKnee_x,LKnee_y,LAnkle_x,LAnkle_y ,REye_x,REye_y,LEye_x,LEye_y ,REar_x,REar_y,LEar_x,Lear_y,class'

def create_dataset():
    sit_fname = 'sit.txt'
    # work_fname = 'work2.txt'
    # stand_fname = 'stand2.txt'
    walk_fname = 'walk.txt'
    lie_fname = 'lie.txt'

    # fnames = [sit_fname, work_fname, stand_fname, walk_fname, lie_fname]
    fnames = [sit_fname, walk_fname, lie_fname]
    header_list = HEADER.replace(' ', '').split(',')
    print(len(header_list))
    data_arrays = []

    for idx, fname in enumerate(fnames):
        print(fname, idx)
        with open(fname) as f:
            lines = f.readlines()
            f.close()

        lines = list(filter(lambda x: x != '\n', lines))
        lines = list(map(lambda x: list(map(lambda y: float(y), x.split(' ')))[:36] + [float(idx)], lines))

        line_array = np.array(lines)

        print(line_array.shape)
        # print(line_array[0, :])
        
        data_arrays.append(line_array)

    
    concated = np.concatenate(data_arrays, axis=0)
    print(concated.shape)

    df = pd.DataFrame(concated, columns=header_list)
    # print(df)
    df.to_csv('lounge_data4.csv', sep= ',', index=False)


if __name__ == '__main__':
    create_dataset()