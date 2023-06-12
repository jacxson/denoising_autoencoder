import os
import librosa
import numpy as np
import soundfile as sf


"""
The style and approach of my preprocessing scripts was heavily influenced by the 
preprocessing scripts in this repository: 

https://github.com/musikalkemist/generating-sound-with-neural-networks/blob/main/12%20Preprocessing%20pipeline/preprocess.py
"""


class Splitter:
    
    def __init__(self, target_sr, target_length):
        
        self.target_sr = target_sr
        self.target_length = target_length

        
    def process_audio_files(self, directory, save_dir):
        
        for audio_file in os.listdir(directory):
            if 'DS_Store' not in audio_file:
                file_path = os.path.join(directory, audio_file)
                signal, sr = self._load_audio_file(file_path)
                if self._is_resample_necessary(sr):
                    signal = self._resample_audio(signal, sr)
                audio_slices = self._split_audio_file(signal)
                self._save_audio_slices(audio_slices, audio_file, save_dir)
        print(f"""
        {len(os.listdir(save_dir))} {self.target_length / self.target_sr}-second audio files created in:
        {save_dir}
        """)
        
        
    def _load_audio_file(self, file_path):
        
        signal, sr = librosa.load(file_path)
        return signal, sr
        
        
    def _resample_audio(self, signal, sr):
        
        resampled_signal = librosa.resample(signal, orig_sr=sr, 
                                            target_sr=self.target_sr)
        return resampled_signal
        
        
    def _is_resample_necessary(self, sr):
        
        if sr != self.target_sr:
            return True
        else:
            return False
    
    
    def _split_audio_file(self, signal):
        """
        Adapted from this implementation of splitting wav files with librosa:
        https://stackoverflow.com/a/60115003
        """
        signal_len = len(signal)
        slice_start = 0
        threshold = self.target_length / 2
        audio_slices = []
        
        while slice_start < signal_len - threshold:
            
            slice_end = slice_start + self.target_length
            if slice_end > signal_len:
                audio_slice = signal[slice_start:]
                audio_slice = self._apply_padding(audio_slice)
            else:
                audio_slice = signal[slice_start:slice_end]
            
            audio_slices.append(audio_slice)
            slice_start = slice_end
        return audio_slices
    
    
    def _apply_padding(self, audio_slice): 
        """
        Adapted from https://github.com/musikalkemist/generating-sound-with-
        neural-networks/blob/main/12%20Preprocessing%20pipeline/preprocess.py
        """
        padding_size = self.target_length - len(audio_slice)
        padded_signal = self._right_pad(audio_slice, padding_size)
        return padded_signal
    
    
    def _right_pad(self, audio_slice, padding_size):
        
        padded_signal = np.pad(audio_slice, (0, padding_size), mode="constant")
        return padded_signal
    
    
    def _save_audio_slices(self, audio_slices, original_file_name, save_dir):
        
        for i, audio_slice in enumerate(audio_slices):
            
            file_name_without_extension = original_file_name.split(".")[0]
            save_file_name = file_name_without_extension + f"_split_{i + 1}.wav"
            
            if not os.path.exists(f"{save_dir}"):
                os.makedirs(save_dir)
                
            save_file_path = os.path.join(save_dir, save_file_name)
            sf.write(save_file_path, audio_slice, self.target_sr)
    