import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import random
from pathlib import Path
from collections import defaultdict
import sys


def process_google_sample(signal, frame_length=0.04, frame_shift=0.02, n_mfcc=40):
    """Process audio sample to extract MFCC features."""
    sr = 16000
    if len(signal) != 16000:
        return None

    frame_length_samples = int(frame_length * sr)
    frame_shift_samples = int(frame_shift * sr)

    signal = (signal - signal.mean()) / signal.std()

    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=2048, n_mfcc=n_mfcc, 
                                hop_length=frame_shift_samples,
                                win_length=frame_length_samples,
                                center=True, norm=None)
    mfccs = np.transpose(mfccs)[1:-1]
    return mfccs

class AudioMixer:
    def __init__(self, gsc_dir, esc50_dir, output_dir, target_sr=16000):
        self.gsc_dir = Path(gsc_dir)
        self.esc50_dir = Path(esc50_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GSC classes
        self.commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        self.unknown = ['eight', 'nine', 'bed', 'five', 'dog', 'seven', 'three', 'marvin', 'wow',
                       'house', 'sheila', 'two', 'six', 'cat', 'one', 'tree', 'four', 'happy', 'zero', 'bird']
        
        # ESC-50 categories mapping
        self.esc_categories = {
            0: 'dog', 1: 'rooster', 2: 'pig', 3: 'cow', 4: 'frog',
            5: 'cat', 6: 'hen', 7: 'insects', 8: 'sheep', 9: 'crow',
            10: 'rain', 11: 'sea_waves', 12: 'crackling_fire', 13: 'crickets',
            14: 'chirping_birds', 15: 'water_drops', 16: 'wind', 17: 'pouring_water',
            18: 'toilet_flush', 19: 'thunderstorm', 20: 'crying_baby', 21: 'sneezing',
            22: 'clapping', 23: 'breathing', 24: 'coughing', 25: 'footsteps',
            26: 'laughing', 27: 'brushing_teeth', 28: 'snoring', 29: 'drinking_sipping',
            30: 'door_wood_knock', 31: 'mouse_click', 32: 'keyboard_typing',
            33: 'door_wood_creaks', 34: 'can_opening', 35: 'washing_machine',
            36: 'vacuum_cleaner', 37: 'clock_alarm', 38: 'clock_tick',
            39: 'glass_breaking', 40: 'helicopter', 41: 'chainsaw', 42: 'siren',
            43: 'car_horn', 44: 'engine', 45: 'train', 46: 'church_bells',
            47: 'airplane', 48: 'fireworks', 49: 'hand_saw'
        }
        
        # Initialize ESC-50 files by category
        self.esc_files_by_category = defaultdict(list)
        self._init_esc_files()

    def _init_esc_files(self):
        """Initialize ESC-50 files by category."""
        for esc_file in self.esc50_dir.glob('*.wav'):
            category = self.get_esc_category(esc_file)
            self.esc_files_by_category[category].append(esc_file)

    def get_class_label(self, file_path):
        """Get the class label for a GSC audio file."""
        class_name = file_path.parent.name
        
        if class_name in self.commands:
            return self.commands.index(class_name)
        elif class_name == '_background_noise_':
            return 11
        else:
            return 10  # unknown

    def load_audio(self, file_path):
        """Load audio file and resample if necessary."""
        audio, sr = librosa.load(file_path, sr=self.target_sr)
        return audio

    def adjust_snr(self, signal, noise, target_snr):
        """Adjust noise to achieve target SNR."""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Handle zero power cases
        if signal_power == 0 or noise_power == 0:
            return np.zeros_like(noise)
            
        # Calculate scale factor
        scale_factor = np.sqrt(signal_power / (noise_power * 10 ** (target_snr / 10)))
        
        # # 调试输出 SNR 相关参数
        # print(f"[adjust_snr] target_snr: {target_snr} dB, signal_power: {signal_power}, noise_power: {noise_power}, scale_factor: {scale_factor}")
        
        # Apply scaling
        adjusted_noise = noise * scale_factor
        
        # Check for invalid values
        if not np.all(np.isfinite(adjusted_noise)):
            return np.zeros_like(noise)
            
        return adjusted_noise

    def mix_audio(self, gsc_audio, esc_audio, snr):
        """Mix GSC audio with ESC-50 audio at specified SNR."""
        # Ensure inputs are valid
        if not np.all(np.isfinite(gsc_audio)) or not np.all(np.isfinite(esc_audio)):
            return gsc_audio
            
        # Adjust ESC audio length to match GSC audio length
        gsc_len = len(gsc_audio)
        esc_len = len(esc_audio)
        
        if esc_len < gsc_len:
            # If ESC is shorter, repeat it to match GSC length
            repeats = gsc_len // esc_len
            remainder = gsc_len % esc_len
            esc_audio = np.tile(esc_audio, repeats)
            if remainder > 0:
                esc_audio = np.concatenate([esc_audio, esc_audio[:remainder]])
        else:
            # If ESC is longer, truncate it to GSC length
            esc_audio = esc_audio[:gsc_len]
        
        # Adjust noise to achieve target SNR
        adjusted_noise = self.adjust_snr(gsc_audio, esc_audio, snr)
        
        # Mix the audio
        mixed_audio = gsc_audio + adjusted_noise
        
        # Normalize
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0:
            mixed_audio = mixed_audio / max_val
        
        # Final check for invalid values
        if not np.all(np.isfinite(mixed_audio)):
            return gsc_audio
        
        return mixed_audio

    def get_esc_category(self, esc_filename):
        """Get ESC-50 category from filename."""
        # Parse filename: {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
        target = int(esc_filename.stem.split('-')[-1])
        return target

    def process_dataset(self, snr_db=10, esc_category=None):
        """Process test dataset with mixed ESC-50 audio."""
        if esc_category is None:
            raise ValueError("ESC category must be specified")
            
        # Read test list
        with open(self.gsc_dir / 'testing_list.txt', 'r') as f:
            test_list = [line.strip() for line in f]
        test_set = set(test_list)
        
        # Get ESC category name
        esc_category_name = self.esc_categories[esc_category]
        
        # Create category and SNR directory
        category_dir = self.output_dir / 'test' / esc_category_name / f'{snr_db}db'
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counter for file naming
        cnt = 0
        
        # Process command words
        print('Processing command words...')
        for i, class_name in enumerate(tqdm(self.commands)):
            sample_dir = self.gsc_dir / class_name
            if not sample_dir.exists():
                continue
                
            for sample in sample_dir.glob('*.wav'):
                rel_path = str(sample.relative_to(self.gsc_dir))
                
                if rel_path in test_set:
                    # Load and process audio
                    signal = self.load_audio(sample)
                    
                    # Get ESC file
                    esc_files = self.esc_files_by_category[esc_category]
                    if esc_files:
                        esc_file = random.choice(esc_files)
                        esc_audio = self.load_audio(esc_file)
                        
                        # Mix audio
                        mixed_audio = self.mix_audio(signal, esc_audio, snr_db)
                        
                        # Extract MFCC from mixed audio
                        mixed_mfccs = process_google_sample(mixed_audio)
                        if mixed_mfccs is not None:
                            data = {'feature': mixed_mfccs, 'label': i}
                            np.save(category_dir / f'{cnt}.npy', data)
                            cnt += 1
        
        # Process unknown words
        print('Processing unknown words...')
        for class_name in tqdm(self.unknown):
            sample_dir = self.gsc_dir / class_name
            if not sample_dir.exists():
                continue
                
            for sample in sample_dir.glob('*.wav'):
                rel_path = str(sample.relative_to(self.gsc_dir))
                
                if rel_path in test_set:
                    # Load and process audio
                    signal = self.load_audio(sample)
                    
                    # Get ESC file
                    esc_files = self.esc_files_by_category[esc_category]
                    if esc_files:
                        esc_file = random.choice(esc_files)
                        esc_audio = self.load_audio(esc_file)
                        
                        # Mix audio
                        mixed_audio = self.mix_audio(signal, esc_audio, snr_db)
                        
                        # Extract MFCC from mixed audio
                        mixed_mfccs = process_google_sample(mixed_audio)
                        if mixed_mfccs is not None:
                            data = {'feature': mixed_mfccs, 'label': 10}  # unknown class
                            np.save(category_dir / f'{cnt}.npy', data)
                            cnt += 1
        
        # Process silence
        print('Processing silence...')
        sample_dir = self.gsc_dir / '_background_noise_'
        if sample_dir.exists():
            for sample in sample_dir.glob('*.wav'):
                if sample.name == 'README.md':
                    continue
                    
                signal = self.load_audio(sample)
                
                # Generate silence samples for test set
                for _ in range(25):  # 25 silence samples for test set
                    idx = random.randint(0, len(signal) - 16000)
                    signal_tmp = signal[idx:idx + 16000]
                    
                    # Get ESC file
                    esc_files = self.esc_files_by_category[esc_category]
                    if esc_files:
                        esc_file = random.choice(esc_files)
                        esc_audio = self.load_audio(esc_file)
                        
                        # Mix audio
                        mixed_audio = self.mix_audio(signal_tmp, esc_audio, snr_db)
                        
                        # Extract MFCC from mixed audio
                        mixed_mfccs = process_google_sample(mixed_audio)
                        if mixed_mfccs is not None:
                            data = {'feature': mixed_mfccs, 'label': 11}  # silence class
                            np.save(category_dir / f'{cnt}.npy', data)
                            cnt += 1
        
        print(f'Finished. Total mixed test samples: {cnt}')

def get_mean_std(data_dir):
    mean_list=[]
    std_list=[]
    for i in tqdm(os.listdir(data_dir)):
        data=np.load(data_dir+'/'+i,allow_pickle=True).tolist()
        feature=data['feature']
        mean_list.append(np.mean(feature,axis=0))
        std_list.append(np.std(feature,axis=0))
    mean=np.mean(np.array(mean_list),axis=0)
    std=np.mean(np.array(std_list),axis=0)
    return mean,std

def pre_process(mean, std, data_dir):
    for i in tqdm(os.listdir(data_dir)):
        data=np.load(data_dir+'/'+i,allow_pickle=True).tolist()
        data['feature']=(data['feature']-mean)/std
        np.save((data_dir+'/'+i), data)


def main():
    # Initialize AudioMixer
    mixer = AudioMixer(
        gsc_dir='speech_commands',
        esc50_dir='esc50',
        output_dir='processed_dataset'
    )
    
    # Define ESC categories and SNR levels
    esc_categories = [0, 5, 17, 19, 20, 26, 35, 36, 43, 48]
    snr_db = [-20, -15]
    for esc_category in esc_categories:
        for snr in snr_db:
            print(f"Processing {esc_category} with SNR {snr} dB")
            mixer.process_dataset(snr_db=snr, esc_category=esc_category)
            mean, std = get_mean_std(f'{mixer.output_dir}/test/{mixer.esc_categories[esc_category]}/{snr}db')
            pre_process(mean, std, f'{mixer.output_dir}/test/{mixer.esc_categories[esc_category]}/{snr}db')

if __name__ == "__main__":
    main() 