# encoding=utf-8
import numpy as np
import os
import librosa
from multiprocessing import Pool
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

SR = 44100
class EML():
    def __init__(self, silence_threshold = 2e-2):
        self.silence_threshold = silence_threshold

        # 原始文件设置
        self.data_path = os.path.join('..', 'data')
        self.ext = 'avi'

        # 保存文件设置
        self.save_path_type = 1
        self.save_vedio_dir = os.path.join('..','data1')
        self.save_ext = 'mp4'
        self.save_csv_path = os.path.join(self.data_path, 'data_meta_info.csv')
        self.csv_header = ['path', 'time_len',
                           'available_time_L', 'available_time_R']


    def move_vedio_file(self,path):
        if not os.path.exists(self.save_vedio_dir):
            os.makedirs(self.save_vedio_dir)
        for root, _, names in os.walk(path):
            for name in names:
                if name.endswith(self.ext):
                    video_path = os.path.join(root, name)
                    video_split = video_path.split(os.path.sep)
                    target_name = os.path.join(self.data_path,
                                               '_'.join(video_split[-3:]))
                    os.rename(video_path,target_name)
                    # shutil.copyfile(video_path,target_name)

    def get_audio_list(self,path):
        '''
        获取文件列表
        :param path: str,文件所在的主目录
        :param ext:  str,所要获取的文件后缀名
        :return: list,主目录下所部文件名的列表
        '''
        audio_list = []
        for root, _, names in os.walk(path):
            for name in names:
                if name.endswith(self.ext):
                    audio_list.append(os.path.join(root, name))
        return audio_list

    # 移除音频文件没有声音的区间
    def trim_silence(self,y):
        def trim_left(y):
            l = 0
            for i in np.arange(1000, len(y), 1):
                if y[i] > self.silence_threshold:
                    break
                l = i
            return l

        def trim_right(y):
            r = 0
            for i in np.arange(len(y) - 1, -1, -1):
                if y[i] > self.silence_threshold:
                    break
                r = i
            return r

        left_idx = trim_left(y)
        right_idx = trim_right(y)
        return left_idx, right_idx

    #获取一个音频文件的路径，subject，类别，时长，有效时间间隔
    def get_one_audio_info(self,audio_path):
        audio_split = audio_path.split(os.path.sep)
        csv_target_name = audio_path.split(os.path.sep)[-1]
        print(audio_split)

        audio, sr = librosa.core.load(audio_path)
        csv_time = librosa.get_duration(filename=audio_path)
        L, R = self.trim_silence(audio)
        csv_available_time_L = librosa.samples_to_time(L)[0]
        csv_available_time_R = librosa.samples_to_time(R)[0]
        csv_data = [csv_target_name, csv_time,
                    csv_available_time_L, csv_available_time_R]
        return csv_data

    def get_EML_csv(self):
        audio_lists = self.get_audio_list(self.data_path)
        pool = Pool()
        res = pool.map(self.get_one_audio_info, audio_lists)
        data = pd.DataFrame(data=res)
        data.to_csv(self.save_csv_path, index=False, header=self.csv_header)
        print('数据条数为:{0}'.format(len(res)))

    def get_one_video(self,vedio_path, L, R):
        vedio = VideoFileClip(vedio_path,audio_fps=SR).subclip(L,R)
        vedio_path.replace(self.data_path,self.save_vedio_dir)
        tmp_list = vedio_path.split(os.path.sep)
        tmp_list[-1] = tmp_list[-1].replace(self.ext,self.save_ext)

        if self.save_path_type == 1:
            target_path = os.path.join(self.save_vedio_dir,tmp_list[-1])
        if self.save_path_type == 2:
            target_path = os.path.join(*tmp_list)
            if not os.path.exists(target_path[:-7]):
                os.makedirs(target_path[:-7])
        # print(target_path)
        '''if _WaitForSingleObject(self._handle, 0) == _WAIT_OBJECT_0:
           OSError: [WinError 6] The handle is invalid
           解决方案：https://stackoverflow.com/questions/43966523/getting-oserror-winerror-6-the-handle-is-invalid-in-videofileclip-function
        '''
        vedio.write_videofile(target_path,progress_bar=False,verbose=False)
        vedio.reader.close()
        vedio.audio.reader.close_proc()

    def plot_audio(self,path):
        fig, axs = plt.subplots(2, 1)
        y, sr = librosa.core.load(path)
        L, R = self.trim_silence(y)
        axs[0].plot(y)
        axs[0].set_title(path.split(os.path.sep)[-1])
        y1 = np.zeros(len(y))
        y1[L:R] = y[L:R]
        axs[1].plot(y1)
        plt.show()


# 手工调
def handwork(i,silence_threshold=1e-2):
    eml = EML()
    data_pd = pd.read_csv(eml.save_csv_path)
    path = os.path.join(eml.data_path, data_pd.loc[i]['path'])
    L = data_pd.loc[i]['available_time_L']
    R = data_pd.loc[i]['available_time_R']
    audio,sr = librosa.load(path,sr=SR)
    vedio = VideoFileClip(path,audio_fps=sr).subclip(L, R)
    # print(data_pd.loc[i]['path'])
    eml.plot_audio(path)
    eml.silence_threshold = silence_threshold
    if L < R:
        print(vedio.duration,
              librosa.get_duration(filename=path,sr=sr),
        librosa.get_duration(y=audio,sr=sr))
        eml.get_one_video(path, L, R)

# 初步预处理
def batch_processing():
    eml = EML()
    data_pd = pd.read_csv(eml.save_csv_path)
    count = 0
    for i in range(720):
        path = os.path.join(eml.data_path, data_pd.loc[i]['path'])
        L = data_pd.loc[i]['available_time_L']
        R = data_pd.loc[i]['available_time_R']
        time_len = data_pd.loc[i]['time_len']
        if L < R:
            vedio = VideoFileClip(path,audio_fps=SR).subclip(L,R)
            if np.abs(time_len-vedio.duration) <0.3: #可能删除
                eml.get_one_video(path,L,R)
                count+=1
            else:
                print(path, time_len,vedio.duration,L, R)
            vedio.reader.close()
            vedio.audio.reader.close_proc()
        else:
            print(path,L,R)
            with open(os.path.join(eml.data_path,'d.txt'),'a') as f:
                f.write(path)
                f.write('\n')
    print('需进一步处理的数据条数:{0},保存在文件{1}'.format(count,'data.txt'))

if __name__ == '__main__':
    # eml = EML()
    # eml.move_vedio_file(eml.data_path)
    # eml.get_EML_csv()
    handwork(i=210,silence_threshold=2e-2)
    # batch_processing()

    # "with clip duration=%d seconds, " % self.duration)
    # OSError: Error in file..\data\s4_f3chi_an1.avi, Accessing
    # time
    # t = 3.24 - 3.28
    # seconds,
    # with clip duration=3 seconds,



