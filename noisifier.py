import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os




class Noisifier:
    
    def __init__(self, 
                 audio_dir, 
                 noise_dir, 
                 save_dir, 
                 sr,
                 snr_range =[-5, 5], 
                 seed=123):
        
        self.audio_dir = audio_dir
        self.noise_dir = noise_dir
        self.save_dir = save_dir
        self.sr = sr
        self.snr_min = snr_range[0]
        self.snr_max = snr_range[1]
        self.file_paths_dict = {}
        self.audio_file_names = None
        self.audio_paths = None
        self.noise_file_names = None
        self.noise_paths = None
        self.mixed_file_names = None
        self.mixed_paths = None
        self.metadata_save_path = None
        np.random.seed(seed)
        
        
    
    def noisify(self):
        
        self._create_save_dir()
        
        file_number = 1
        self._get_file_names()
        for audio_file in self.audio_paths:
            self._process_file(audio_file, file_number)
            file_number += 1
        self._save_metadata()
    
    def _create_save_dir(self):
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def _process_file(self, audio_file, file_number):
        
        noise_file = self._get_random_noise_file()
        signal, noise = self._load_files(audio_file, noise_file)
        snr = self._get_random_snr()
        mixed_signal = self.add_noise_to_signal(signal, noise, snr)
        mixed_name, mixed_path = self._get_save_path(file_number)
        self._save_mixed_signal(mixed_signal, mixed_path)
        self._store_metadata(mixed_name, mixed_path, audio_file, 
                             noise_file, snr)
        
    def _get_file_names(self):
        
        self.audio_file_names = os.listdir(self.audio_dir)
        self.audio_paths = [os.path.join(self.audio_dir, x)\
                            for x in self.audio_file_names]
        
        self.noise_file_names = os.listdir(self.noise_dir)
        self.noise_paths = [os.path.join(self.noise_dir, x)\
                            for x in self.noise_file_names]
        
    def _get_random_snr(self):
        
        return np.random.randint(low = self.snr_min, high=self.snr_max)
    
    def _get_random_noise_file(self):
        
        return np.random.choice(self.noise_paths, size=1).item()
    
    def add_noise_to_signal(self, signal, noise, snr):
        """
        Adapted from: https://stackoverflow.com/a/72124325
        """

        noise = noise[np.arange(len(signal)) % len(noise)]

        signal_energy = np.mean(signal**2)
        noise_energy = np.mean(noise**2)

        gain = np.sqrt(10.0**(-snr/10) * signal_energy/(noise_energy + 1e-10))

        signal_coef = np.sqrt(1 / (1 + gain**2))
        noise_coef = np.sqrt(gain**2 / (1 + gain**2))

        return signal_coef * signal + noise_coef * noise
    
    def _load_files(self, audio_file, noise_file):
        
        signal, _ = librosa.load(audio_file)
        noise, _ = librosa.load(noise_file)
        
        return signal, noise
    

    def _get_save_path(self, file_number):
        
        file_name = str(file_number) + '.wav'
        file_path = os.path.join(self.save_dir, file_name)       
        
        return file_name, file_path
    
    def _save_mixed_signal(self, mixed_signal, save_path):
        
        sf.write(save_path, mixed_signal, samplerate=self.sr)
        
    def _store_metadata(self, mixed_name, mixed_path, 
                        audio_file, noise_file, snr):
        
        self.file_paths_dict[mixed_name] = {
            'mixed_file': mixed_path,
            'orig_file': audio_file,
            'noise_file': noise_file,
            'snr': snr
        }
        
    def _save_metadata(self):
        
        if not self.metadata_save_path:
            csv_name = input("Please specify a name for the metadata csv file")
            self.metadata_save_path = f"./audio/{csv_name}.csv" 
            
        metadata = pd.DataFrame(self.file_paths_dict).T
        metadata.to_csv(self.metadata_save_path)
        print(f"Metadata saved at location: {self.metadata_save_path}")