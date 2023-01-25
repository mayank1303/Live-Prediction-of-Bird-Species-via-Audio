import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from librosa import core, onset, feature, display
import soundfile as sf
import umap
from IPython.display import Audio
import sklearn
from sklearn import naive_bayes

#upload all dataframes
df = pd.read_csv("/home/megha/Desktop/Audio_website/templates/audiofiles.csv")
#df.head()
new_dataset = pd.read_csv("/home/megha/Desktop/Audio_website/templates/train.csv").drop(['Unnamed: 0'],axis=1)
#new_dataset.head()

#extracting sr and audi mask from test audio
def load_audio(file_id):
    signal, samplerate =  librosa.load("/home/megha/Desktop/Audio_website/static/uploads/"+str(file_id),duration=50, sr=44100)
    #print(samplerate)
    pre_emphasis = 0.97
    data = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    s = len(data)/samplerate
    sg = librosa.feature.mfcc(y=data,sr=samplerate,
                                     S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)     
    centerpoint = np.argmax(sg.mean(axis=0))
    M = sg[:,centerpoint].mean()
    #print(M)
    mask = sg.mean(axis=0)

    audio_mask = np.zeros(len(data), dtype=bool)
    for i in range(0,len(mask)):
        audio_mask[i*1024:] = mask[i]
    return sg, mask, data, audio_mask, samplerate

def myfunc():
    return lambda x: int(x/6.144000e+03)

#uploading test audio
def main(aud):
    waves = {}
    sg, mask, data, audio_mask, sample_rate = load_audio(str(aud))
    waves['audio'] = data[audio_mask]
    length = len(data[audio_mask])

    w = myfunc()
    windo = w(length)

    windows = {}
    wave = waves['audio']
    species ='gens_specie'
    windows[species] = []
    for i in range(0, int(len(wave)/6.144000e+03)):
        windows[species].append(wave[i:int(i+6.144000e+03)])

	#creating df for test audio
    new_dataset_test = pd.DataFrame()
    for species in windows.keys():
        for i in range(0,len(windows)):
            data_point = {'species':species.split('_')[1], 'genus':species.split('_')[0]}
            #print(type(data_point))
            spec_centroid = feature.spectral_centroid(windows[species][i])[0]
            #print(windows_fixed[species][i])
            chroma = feature.chroma_stft(windows[species][i], sample_rate)
            for j in range(0,13):
                data_point['spec_centr_'+str(j)] = spec_centroid[j]
                for k in range(0,12):
                    data_point['chromogram_'+str(k)+"_"+str(j)] = chroma[k,j]
            new_dataset_test = new_dataset_test.append(data_point,ignore_index=True)


    #classification of test audio
    features= list(new_dataset.columns)
    features.remove('species')
    features.remove('genus')

    X = new_dataset[features].values
    y = new_dataset['species'].values
    X_test = new_dataset_test[features].values
    y_test = new_dataset_test['species'].values

    NB = naive_bayes.GaussianNB()
    SSS = sklearn.model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.2)

    for train_index, val_index in SSS.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
    
        NB.fit(X_train, y_train)    
        y_pred = NB.predict(X_test)
    check = pd.DataFrame()
    df = pd.read_csv("/home/megha/Desktop/Audio_website/templates/descr.csv", delimiter = ';')
    check = df.loc[df['check'] == y_pred[0]]
    #print(check['Description'])
	#accs.append(sklearn.metrics.accuracy_score(y_pred=y_pred, y_true=y_val))    
    return  y_pred[0], check
if __name__ == "__main__":
    result = main()

