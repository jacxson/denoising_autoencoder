import os
import numpy as np
import pandas as pd



class AudioDataset:
    def __init__(self, 
                 sample_directory,
                 metadata_csv,
                 minmax_vals,
                 validation_size, 
                 shuffle=True, 
                 seed=123):
        self.sample_directory = sample_directory
        self.metadata = pd.read_csv(metadata_csv, index_col="Unnamed: 0")
        self.minmax = np.load(minmax_vals, allow_pickle=True).item()
        self.validation_size = int(-1 * validation_size)
        self.seed = seed
        self.shuffle = shuffle
        self.shuffled_index = None
        self.x_train = None
        self.y_train = None
        self.train_min_vals = None
        self.train_max_vals = None
        #self.train_phi None
        self.x_test = None
        self.y_test = None
        self.test_min_vals = None
        self.test_max_vals = None
        #self.test_phi = None
        self.x_train_file_names = None
        self.y_train_file_names = None
        self.x_test_file_names = None
        self.y_test_file_names = None
        
        
        
    def load_data(self):
        
        paths = [self.sample_directory + file_name\
                 for file_name in os.listdir(self.sample_directory)]
        X = self._X_from_file_list(paths)
        if self.shuffle:
            self.shuffled_index = self._shuffle_X(X)
        self._train_test_split(X, paths)
        self._create_y()
        self._get_origninal_minmax()
        
    def _X_from_file_list(self, file_paths):
        
        X = []
        for i in range(len(file_paths)):
            spectrogram = np.load(file_paths[i], allow_pickle=True)
            if spectrogram.shape == (512, 256, 2):
                X.append(spectrogram[...,0])
            elif spectrogram.shape == (512, 256):
                X.append(spectrogram)
            else:
                np.pad(X, X.shape[1], mode='constant')
        X = np.stack(X)
        X_reshaped = X[..., np.newaxis]
        
        return X_reshaped
    
    def _train_test_split(self, X, file_path_list):
        
        train_index = self.shuffled_index[:self.validation_size]
        val_index = self.shuffled_index[self.validation_size:]
        file_paths = np.array(file_path_list)
        
        self.x_train = X[train_index]
        self.x_train_file_names = file_paths[train_index]
        self.x_test = X[val_index]
        self.x_test_file_names = file_paths[val_index]
        
    def _create_y(self):
        
        train_files = [os.path.split(x)[-1][:-4] for x in self.x_train_file_names]
        self.y_train_file_names = np.array([self.metadata.loc[x, 'orig_spec'] for x in train_files])
        self.y_train = np.stack([np.load(x) for x in self.y_train_file_names])[...,np.newaxis]
        #self.train_phi = [x[...,1] for x in orig_train_specs]
        #self.y_train = [x[...,0] for x in orig_train_specs]
        
        test_files = [os.path.split(x)[-1][:-4] for x in self.x_test_file_names]
        self.y_test_file_names = np.array([self.metadata.loc[x, 'orig_spec'] for x in test_files])
        self.y_test = np.stack([np.load(x) for x in self.y_test_file_names])[...,np.newaxis]
        #self.test_phi = [x[...,1] for x in orig_test_specs]
        #self.y_test = [x[...,0] for x in orig_test_specs] 
        
    def _get_origninal_minmax(self):
        
        self.train_min_vals = np.array([self.minmax[x]['min'] for x in self.x_train_file_names])
        self.train_max_vals = np.array([self.minmax[x]['max'] for x in self.x_train_file_names])
        
        self.test_min_vals = np.array([self.minmax[x]['min'] for x in self.x_test_file_names])
        self.test_max_vals = np.array([self.minmax[x]['max'] for x in self.x_test_file_names])
        pass
    def _shuffle_X(self, X):
        
        np.random.seed(self.seed)
        
        index = np.array([*range(len(X))])
        np.random.shuffle(index)
        return index