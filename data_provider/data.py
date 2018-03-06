#encoding=utf-8
from librosa.feature import melspectrogram
import librosa
from librosa.core import load
from librosa import logamplitude
import numpy as np
import os
import csv
from multiprocessing import Pool


def window(y,win_len = 64,stride = 34):
    y_len = y.shape[1]
    print(y_len)
    for i in range(0,y_len,stride):
        if (i+win_len)<y_len:
            yield i,i+win_len


# 移除音频文件没有声音的区间
def trim_silence(y):
    silence_threshold = 5e-2
    def trim_left(y):
        l = 0
        for i in np.arange(1000, len(y), 1):
            if y[i] > silence_threshold:
                break
            l = i
        return l

    def trim_right(y):
        r = 0
        for i in np.arange(len(y) - 1, -1, -1):
            if y[i] > silence_threshold:
                break
            r = i
        return r
    left_idx = trim_left(y)
    right_idx = trim_right(y)
    return left_idx,right_idx

def get_audio_list(path,ext = 'avi'):
    import os
    audio_list = []
    for root,_,names in os.walk(path):
        for name in names:
            if name.endswith(ext):
                audio_list.append(os.path.join(root,name))
    return audio_list


def get_one_audio_info(audio_path):
    audio_split = audio_path.split(os.path.sep)
    csv_target_name = audio_path
    print(audio_split)
    csv_subject = audio_split[2]
    csv_label = audio_split[-1][:2]

    audio,sr = load(audio_path)
    csv_time = librosa.get_duration(audio)
    L,R = trim_silence(audio)
    csv_available_time_L = librosa.samples_to_time(L)[0]
    csv_available_time_R = librosa.samples_to_time(R)[0]
    csv_data = [csv_target_name,csv_subject,csv_label,csv_time,
                        csv_available_time_L,csv_available_time_R]
    return csv_data

def get_EML_csv():
    audio_lists = get_audio_list(os.path.join('..', 'data'))
    pool = Pool()
    res = pool.map(get_one_audio_info, audio_lists)
    header = ['path', 'subject', 'class','time_len',
              'available_time_L','available_time_R']
    import pandas as pd
    data = pd.DataFrame(data=res)
    data.to_csv('data.csv', index=False, header=header)
    print(len(res))

def get_one_video(vedio_path,L,R,save_pt='data1'):
    from moviepy.editor import VideoFileClip
    vedio = VideoFileClip(vedio_path).subclip(L,R)
    # vedio.write_videofile(os.path.join(save_pt,vedio_path.replace('avi','mp4')))
    vedio.write_videofile(save_pt+'.mp4')


def plot_audio(path):
    import matplotlib.pyplot as plt
    _,axs = plt.subplots(2,1)
    y,sr = load(path)
    L,R = trim_silence(y)
    axs[0*2].plot(y)
    y1 = np.zeros(len(y))
    y1[L:R] = y[L:R]
    axs[0*2+1].plot(y1)
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    data_pd = pd.read_csv('data.csv')
    # print(data_pd.head(3))
    for i in range(3):
        print(data_pd.loc[i]['path'])
        get_one_video(data_pd.loc[i]['path'],data_pd.loc[i]['available_time_L'],
                      data_pd.loc[i]['available_time_R'],str(i))


