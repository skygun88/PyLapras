import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

HEADER = 'nose_x,nose_y,neck_x,neck_y,Rshoulder_x,Rshoulder_y,Relbow_x ,Relbow_y,Rwrist_x,RWrist_y ,LShoulder_x,LShoulder_y,LElbow_x,LElbow_y,LWrist_x,LWrist_y ,RHip_x,RHip_y,RKnee_x,RKnee_y,RAnkle_x,RAnkle_y,LHip_x,LHip_y,LKnee_x,LKnee_y,LAnkle_x,LAnkle_y ,REye_x,REye_y,LEye_x,LEye_y ,REar_x,REar_y,LEar_x,Lear_y,pred,y_class'

def create_result():
    sit_fname = 'result/sit_test2.txt'
    work_fname = 'result/work_test2.txt'
    stand_fname = 'result/stand_test2.txt'
    walk_fname = 'result/walk_test2.txt'
    lie_fname = 'result/lie_test2.txt'

    fnames = [sit_fname, work_fname, stand_fname, walk_fname, lie_fname]
    header_list = HEADER.replace(' ', '').split(',')
    print(len(header_list))
    data_arrays = []

    for idx, fname in enumerate(fnames):
        print(fname, idx)
        with open(fname) as f:
            lines = f.readlines()
            f.close()

        lines = list(filter(lambda x: x != '\n', lines))
        lines = list(map(lambda x: list(map(lambda y: float(y), x.split(' ')))[:37] + [float(idx)] if len(x) > 10 else [0.0]*36 + [float(x), float(idx)], lines))

        line_array = np.array(lines)

        print(line_array.shape)
        # print(line_array[0, :])
        
        data_arrays.append(line_array)

    
    concated = np.concatenate(data_arrays, axis=0)
    print(concated.shape)

    df = pd.DataFrame(concated, columns=header_list)
    # print(df)
    df.to_csv('result.csv', sep= ',', index=False)

def analysis():
    df = pd.read_csv('result.csv', header=0)
    labels = ['sit', 'work', 'stand', 'walk', 'lie', 'overall']
    lens = [len(df[df.y_class == i]) for i in range(5)]
    
    no_detects = [(df[df.y_class == i]['pred'] == -1).sum() for i in range(5)]
    
    filtered_df = df[df.pred != -1]

    tps = [(df[(df.y_class == i) & (df.pred > -1)]['pred'] == i).sum() for i in range(5)]
    fns = [(df[(df.y_class == i) & (df.pred > -1)]['pred'] != i).sum() for i in range(5)]
    tns = [(df[(df.y_class != i) & (df.pred > -1)]['pred'] != i).sum() for i in range(5)]
    fps = [(df[(df.y_class != i) & (df.pred > -1)]['pred'] == i).sum() for i in range(5)]
    precisions = [tps[i]/(tps[i]+fps[i]) for i in range(5)]
    recalls = [tps[i]/(tps[i]+fns[i]) for i in range(5)]
    fmeasures = [2*precisions[i]*recalls[i]/(precisions[i]+recalls[i]) for i in range(5)]
    acces = [(tps[i]+tns[i])/(tps[i]+tns[i]+fns[i]+fps[i]) for i in range(5)]

    all_precision = sum(tps)/(sum(tps)+sum(fps))
    all_recall = sum(tps)/(sum(tps)+sum(fns))
    all_fmeasure = (2*all_precision*all_recall)/(all_precision+all_recall)
    all_acc = (filtered_df['pred'] == filtered_df['y_class']).sum()/len(filtered_df)
    all_nodetect = sum(no_detects)/len(df)

    no_detect_rate = [no_detects[i]/lens[i] for i in range(len(lens))]

    missing_rate = [(df[df.y_class == i].iloc[:, :-2] == 0).sum().sum() / df[df.y_class == i].iloc[:, :-2].count().sum() for i in range(5)]
    all_missing = (df.iloc[:, :-2] == 0).sum().sum() / df.iloc[:, :-2].count().sum()
    print(missing_rate, all_missing)

    all_data = [
                acces+[all_acc], 
                precisions+[all_precision], 
                recalls+[all_recall], 
                fmeasures+[all_fmeasure],
                no_detect_rate+[all_nodetect],
                missing_rate+[all_missing]
                ]
    result_df = pd.DataFrame(np.array(all_data), index=['Accuracy', 'Precision', 'Recall', 'F-measure', 'No-detection', 'Missing'], columns=labels)
    print(result_df)
    with pd.ExcelWriter('overall_analysis.xlsx', engine='xlsxwriter') as writer:
        result_df.to_excel(writer, sheet_name='Sheet1')

    # ax = sns.barplot(data=result_df.filter(items=['F-measure'], axis=0))
    # plt.ylabel('F-measure')
    # plt.ylim(0.0, 1.0)
    # plt.show()

    btps = np.array([(df[df.y_class == i]['pred'] == i).sum() for j in range(5) for i in range(5)]).reshape(5,5)
    bfns = np.array([(df[df.y_class == i]['pred'] == j).sum() for j in range(5) for i in range(5)]).reshape(5,5)
    btns = np.array([(df[df.y_class == j]['pred'] == j).sum() for j in range(5) for i in range(5)]).reshape(5,5)
    bfps = np.array([(df[df.y_class == j]['pred'] == i).sum() for j in range(5) for i in range(5)]).reshape(5,5)
    bprecisions = btps/(btps+bfps)
    brecalls = btps/(btps+bfns)
    bfmeasures = (2*bprecisions*brecalls)/(bprecisions+brecalls)
    bacces = (btps+btns)/(btps+btns+bfps+bfns)

    pre_df = pd.DataFrame(bprecisions, index=['sit', 'work', 'stand', 'walk', 'lie'], columns=['sit', 'work', 'stand', 'walk', 'lie'])
    recall_df = pd.DataFrame(brecalls, index=['sit', 'work', 'stand', 'walk', 'lie'], columns=['sit', 'work', 'stand', 'walk', 'lie'])
    fm_df = pd.DataFrame(bfmeasures, index=['sit', 'work', 'stand', 'walk', 'lie'], columns=['sit', 'work', 'stand', 'walk', 'lie'])
    acc_df = pd.DataFrame(bacces, index=['sit', 'work', 'stand', 'walk', 'lie'], columns=['sit', 'work', 'stand', 'walk', 'lie'])


    with pd.ExcelWriter('binary_analysis.xlsx', engine='xlsxwriter') as writer:
        pre_df.to_excel(writer, sheet_name='Precision')
        recall_df.to_excel(writer, sheet_name='Recall')
        fm_df.to_excel(writer, sheet_name='F-measure')
        acc_df.to_excel(writer, sheet_name='Accuracy')

def performance():
    df = pd.read_csv('fps.csv', header=None)
    # print(df)
    min_fps = 1/df.max()
    max_fps = 1/df.min()
    mean_fps = 1/df.mean()
    # print(min_fps, max_fps, mean_fps)
    # print(min_fps)
    # print(df.max()[0])

    result_df = pd.DataFrame(np.array([min_fps[0], mean_fps[0], max_fps[0]]).reshape(1,3), index=['FPS'], columns=['Min', 'Mean', 'Max'])
    # print(result_df)
    ax = sns.barplot(data=result_df)
    plt.ylabel('FPS')
    plt.ylim(0.0, 15)
    plt.show()

if __name__ == '__main__':
    # create_result()
    analysis()
    # performance()