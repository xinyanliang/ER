#encoding=utf-8
from util import audio_util
import numpy as np
import pickle

def load_data(path,n_mfcc = 20):
    from sklearn.preprocessing import LabelEncoder
    import pickle
    import os
    import librosa
    audios = []
    labels = []
    for root,_,names in os.walk(path):
        for audio_path in names:
            audio_path = os.path.join(path,audio_path)
            audio, sr = librosa.load(audio_path)
            mel_spectc = librosa.feature.mfcc(audio,sr, n_mfcc=n_mfcc,
                                              n_fft=512,hop_length=256)
            audios.append(mel_spectc)
            labels.append(audio_path.split(os.path.sep)[-1].split('_')[-1][0:2])
    labels = np.array(labels)
    label_encode =LabelEncoder()
    label_encode.fit(['an','di','fe','ha','sa','su'])
    labels = label_encode.transform(labels)
    labels = labels.astype(np.int8)

    f = open('EML_X_y.pkl', 'wb')
    pickle.dump((audios, labels), f)
    return audios, labels

class Features():
    def __init__(self,data_f='EML_X_y.pkl'):
        self.audios, self.labels = pickle.load(open(data_f, 'rb'))

    def get_mfcc(self):
        X = []
        for i in self.audios:
            x_mean = np.mean(i,axis=1)
            x_median = np.median(i,axis=1)
            x_std = np.std(i,axis=1)
            x_max = np.max(i,axis=1)
            x_min = np.min(i,axis=1)
            x = np.concatenate((x_mean,x_median,x_std,x_max,x_min))
            X.append(x)
        X = np.array(X)
        return X,self.labels

    def get_cor_feats(self,X,n_expands,cor_mat=None,is_norm=True,is_p_test=True):
        from util import couple
        if cor_mat is None:
            cor_mat = couple.get_couped_cor_mat(X, n_expands=n_expands,is_p_test=is_p_test)
        cor_feats = couple.getX_coupled_feas(X, coupled_cor_mat=cor_mat, is_norm=is_norm)
        return cor_feats,cor_mat

    def get_other(self):
        pass
