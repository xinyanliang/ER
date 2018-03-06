# encoding=utf-8
import numpy as np
import os
import librosa
from multiprocessing import Pool
import pandas as pd

class EML():
    def __init__(self, silence_threshold = 5e-2):
        self.silence_threshold = silence_threshold

        # 保存文件设置
        self.save_path_type = 1
        self.save_vedio_dir = 'data1'
        self.save_ext = 'mp4'
        self.csv_header = ['path', 'subject', 'class', 'time_len',
                           'available_time_L', 'available_time_R']

        # 原始文件设置
        self.data_path = os.path.join('..', 'data')
        self.save_csv_path = os.path.join(self.data_path,'data_meta_info.csv')
        self.ext = 'avi'

    def rename_vedio_file(self,path):
        for root, _, names in os.walk(path):
            for name in names:
                if name.endswith(self.ext):
                    video_path = os.path.join(root, name)
                    video_split = video_path.split(os.path.sep)
                    target_name = video_path.replace(video_split[-1],'_'.join(video_split[-3:]))
                    os.rename(video_path,target_name)

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
        csv_target_name = audio_path
        print(audio_split)
        csv_subject = audio_split[2]
        csv_label = audio_split[-1][:2]

        audio, sr = librosa.core.load(audio_path)
        csv_time = librosa.get_duration(audio)
        L, R = self.trim_silence(audio)
        csv_available_time_L = librosa.samples_to_time(L)[0]
        csv_available_time_R = librosa.samples_to_time(R)[0]
        csv_data = [csv_target_name, csv_subject, csv_label, csv_time,
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
        from moviepy.editor import VideoFileClip
        vedio = VideoFileClip(vedio_path).subclip(L, R)
        tmp_list = vedio_path.split(os.path.sep)
        tmp_list[-1] = tmp_list[-1].replace(self.ext,self.save_ext)
        tmp_list[1] = tmp_list[1].replace('data',self.save_vedio_dir)

        if self.save_path_type == 1:
            target_path = os.path.join('..',self.save_vedio_dir,tmp_list[-1])
        if self.save_path_type == 2:
            target_path = os.path.join(*tmp_list)
            if not os.path.exists(target_path[:-7]):
                os.makedirs(target_path[:-7])

        vedio.write_videofile(target_path)

if __name__ == '__main__':
    eml = EML()
    eml.rename_vedio_file(eml.data_path)
    eml.get_EML_csv()
    # import pandas as pd
    # data_pd = pd.read_csv('data.csv')

    # for i in range(3):
    #     print(data_pd.loc[i]['path'])
    #     eml.get_one_video(data_pd.loc[i]['path'],data_pd.loc[i]['available_time_L'],
    #                   data_pd.loc[i]['available_time_R'])
