import os
import ast
import librosa
from cmath import rect
import numpy as np
import pandas as pd




# based on a similar spectrogram extractor found here:
# https://github.com/musikalkemist/generating-sound-with-neural-networks/blob/main/12%20Preprocessing%20pipeline/preprocess.py

class PolarSpectrogramExtractor:
    """
    Extracts 2-Channel Spectrograms from an audio signal using the short time
    fourier transform assigning the log magnitude (db) spectrogram to the first
    channel and the angle in radians to the second channel. 
    """

    def __init__(self, frame_size, 
                 hop_length, 
                 rho_minmax=[-100,100], 
                 theta_minmax=[-3.1415925, 3.1415925], 
                 target_minmax=[0, 1]):
        
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.target_min = target_minmax[0]
        self.target_max = target_minmax[1]
        self.rho_min = rho_minmax[0]
        self.rho_max = rho_minmax[1]
        self.theta_min = theta_minmax[0]
        self.theta_max = theta_minmax[1]
        
    def extract(self, signal):
        
        stft = self.stft(signal)
        rho_spec, theta = self.get_polar_form(stft)
        formatted_spectrogram = self.format_spectrogram(rho_spec, theta)
        norm_spec = self.normalize(formatted_spectrogram)

        return norm_spec
    
    def recover_signal(self, formatted_spectrogram):
        stft = self.recover_stft(formatted_spectrogram)
        signal = self.istft(stft)
        
        return signal
    
    def recover_stft(self, formatted_spectrogram):

        spec = self.denormalize(formatted_spectrogram)
        rho = spec[...,0]
        theta = spec[...,1]
        stft = self.polar_to_stft(rho, theta)
        
        return stft
    
    def stft(self, signal):
        
        spectrogram = librosa.stft(signal,
                                   n_fft=self.frame_size,
                                   hop_length=self.hop_length)[:-1, :-1]
        return spectrogram
    
    def istft(self, spectrogram):
        
        signal = librosa.istft(spectrogram, 
                               n_fft=self.frame_size, 
                               hop_length=self.hop_length)
        
        return signal
    
    def format_spectrogram(self, rho, theta):
        
        formatted_spectrogram = np.stack([rho, theta], axis=2)
        return formatted_spectrogram
            
    def get_polar_form(self, spectrogram):

        rho = np.abs(spectrogram)
        rho = librosa.amplitude_to_db(rho)
        theta = np.angle(spectrogram)

        return rho, theta

    def polar_to_stft(self, rho, theta):

        # from https://stackoverflow.com/a/27788291
        nprect = np.vectorize(rect)
        
        rho = librosa.db_to_amplitude(rho)
        stft = nprect(rho, theta)

        return stft
    
    def normalize(self, spectrogram):
        
        rho = spectrogram[...,0]
        theta = spectrogram[...,1]
        
        norm_rho = self._normalize_rho(rho)
        norm_theta = self._normalize_theta(theta)
        norm_spec = np.stack([norm_rho, norm_theta], axis=2)
        
        return norm_spec
    
    def _normalize_rho(self, rho):
        
        norm_rho = ((rho - self.rho_min) / 
                    (self.rho_max - self.rho_min))
        norm_rho = (norm_rho * (self.target_max - self.target_min) 
                    + self.target_min)
        return norm_rho
    
    def _normalize_theta(self, theta):
        
        norm_theta = ((theta - self.theta_min) / 
                    (self.theta_max - self.theta_min))
        norm_theta = (norm_theta * (self.target_max - self.target_min) 
                    + self.target_min)
        return norm_theta
    
    def denormalize(self, normalized_spectrogram):
        
        norm_rho = normalized_spectrogram[...,0]
        norm_theta = normalized_spectrogram[...,1]
        
        rho = self._denormalize_rho(norm_rho)
        theta = self._denormalize_theta(norm_theta)
        spec = np.stack([rho, theta], axis=2)
        
        return spec
        
    def _denormalize_rho(self, norm_rho):
        
        rho = ((norm_rho - self.target_min) / 
               (self.target_max - self.target_min))
        rho = (rho * (self.rho_max - self.rho_min) + self.rho_min)
        return rho
    
    def _denormalize_theta(self, norm_theta):
        
        theta = ((norm_theta - self.target_min) / 
               (self.target_max - self.target_min))
        theta = (theta * (self.theta_max - self.theta_min) + self.theta_min)
        return theta
    
    

    
class AudioDatasetFromCSV:
    def __init__(self, 
                 spectrogram_extractor, 
                 metadata_csv, 
                 validation_size=0.1, 
                 sr = 22050,
                 prepared = True,
                 seed=123):
        self.spectrogram_extractor = spectrogram_extractor
        self.metadata_csv = metadata_csv
        self.validation_size = validation_size
        self.sr = sr
        np.random.seed(seed)
        self.metadata = None
        self.spectrogram_directory = None
        self.prepared = prepared
    
    def prepare_dataset(self):
        if not self.prepared:
            self._get_metadata()
            self._create_spec_columns()
            self._extract_spectrograms()
            self.metadata.to_csv("./dataset_metadata.csv")
            self.prepared = True
        else:
            self.metadata = pd.read_csv(self.metadata_csv, index_col="Unnamed: 0")
            print(
                """
                Dataset has already been prepared. To finish preparing data for
                training use dataset.train_test_split(). To rerun dataset preparation, 
                you must first set the 'prepared' attribute to False.
                """)
    def train_test_split(self):
    
        train_data, test_data = self._split_train_test_data()
        self._get_train_test_data(train_data, test_data)
        self._get_train_test_files(train_data, test_data)
        self._split_rho_theta()
    
    def _split_rho_theta(self):
        
        self.x_train_rho = self.x_train[...,0][...,np.newaxis]
        self.y_train_rho = self.y_train[...,0][...,np.newaxis]
        self.x_test_rho = self.x_test[...,0][...,np.newaxis]
        self.y_test_rho = self.y_test[...,0][...,np.newaxis]

        self.x_train_theta = self.x_train[...,1][...,np.newaxis]
        self.y_train_theta = self.y_train[...,1][...,np.newaxis]
        self.x_test_theta = self.x_test[...,1][...,np.newaxis]
        self.y_test_theta = self.y_test[...,1][...,np.newaxis]
    
    def _get_train_test_files(self, train_data, test_data):
        
        # train_files
        self.x_train_files = np.array(train_data['mixed_file'])
        self.y_train_files = np.array(train_data['orig_file'])
        self.x_train_noise_files = np.array(train_data['noise_file'])
        
        # test_files
        self.x_test_files = np.array(test_data['mixed_file'])
        self.y_test_files = np.array(test_data['orig_file'])
        self.x_test_noise_files = np.array(test_data['noise_file'])
    
    def _get_train_test_data(self, train_data, test_data):
        
        # load_train_data
        self.x_train = np.stack([np.load(x) for x in 
                                 train_data['mixed_spec']], axis=0)
        self.y_train = np.stack([np.load(y) for y in 
                                 train_data['orig_spec']], axis=0)
        self.x_train_noise = np.stack([np.load(x) for x in 
                                       train_data['noise_spec']], axis=0)
        # load_test_data
        self.x_test = np.stack([np.load(x) for x in 
                                test_data['mixed_spec']], axis=0)
        self.y_test = np.stack([np.load(y) for y in 
                                test_data['orig_spec']], axis=0)
        self.x_test_noise = np.stack([np.load(x) for x in 
                                       test_data['noise_spec']], axis=0)
        
    def _split_train_test_data(self):
        
        val_size = int(np.ceil(len(self.metadata) * self.validation_size))
        shuffled_data = self.metadata.sample(frac=1)
        train_data = shuffled_data[:-val_size]
        test_data = shuffled_data[-val_size:]
        print(f"Length of Training Data: {len(train_data)}")
        print(f"Length of Testing Data: {val_size}")
        return train_data, test_data
        
    
    def _extract_spectrograms(self):
        
        for idx in self.metadata.index:
            mixed, original, noise = self._load_signals(idx)
            self._save_spectrograms(mixed, idx, "mixed_spec")
            self._save_spectrograms(original, idx, "orig_spec")
            self._save_spectrograms(noise, idx, "noise_spec")
        
    def _save_spectrograms(self, signal, idx, spec_col):
        
        spec = self.spectrogram_extractor.extract(signal)
        spec_path = self.metadata.loc[idx, spec_col]
        np.save(spec_path, spec)
        
    def _get_metadata(self):
        self.metadata = pd.read_csv(self.metadata_csv, index_col="Unnamed: 0")
        self._create_spec_columns()
        
    def _create_spec_columns(self):
        
        def get_spec_path(path):
            
            wav_dir, file = os.path.split(path)
            spec_file = file + ".npy"
            spec_dir = wav_dir + "_specs"
            self._create_spec_dir(spec_dir)
            spec_path = os.path.join(spec_dir, spec_file)
            
            return spec_path
            
        for col in self.metadata.columns[:3]:
            spec_col = col.replace("file", "spec")
            self.metadata[spec_col] = self.metadata[col].apply(get_spec_path)
            
    def _create_spec_dir(self, spec_dir):
        
        if not os.path.exists(spec_dir):
            os.makedirs(spec_dir)
            
    def _load_signals(self, idx):
        files = self.metadata.loc[idx,['mixed_file', 
                                        'orig_file', 
                                        'noise_file']]
        
        mixed, _ = librosa.load(files[0], sr=self.sr)
        orig, _ = librosa.load(files[1], sr=self.sr)
        noise, _ = librosa.load(files[2], sr=self.sr)
        
        return mixed, orig, noise