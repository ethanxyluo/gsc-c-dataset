import os
import numpy as np
import librosa
import soundfile as sf
import random
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from collections import defaultdict

def get_esc_category(esc_filename):
    """Get ESC-50 category from filename."""
    # Parse filename: {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
    target = int(esc_filename.stem.split('-')[-1])
    return target

def extract_mfcc(audio, sr=16000):
    """Extract MFCC features from audio."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=int(0.04 * sr),
        hop_length=int(0.02 * sr)
    )
    return mfcc.T

def calculate_snr(signal, noise):
    """Calculate SNR between signal and noise."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if signal_power == 0:
        return -np.inf
    if noise_power == 0:
        return np.inf
        
    return 10 * np.log10(signal_power / noise_power)

def adjust_snr(signal, noise, target_snr):
    """Adjust noise to achieve target SNR."""
    current_snr = calculate_snr(signal, noise)
    if current_snr == -np.inf or current_snr == np.inf:
        return np.zeros_like(noise)
        
    # Calculate scaling factor
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if signal_power == 0 or noise_power == 0:
        return np.zeros_like(noise)
        
    scaling_factor = np.sqrt(signal_power / (noise_power * 10 ** (target_snr / 10)))
    adjusted_noise = noise * scaling_factor
    
    # Safety check
    if not np.all(np.isfinite(adjusted_noise)):
        return np.zeros_like(noise)
        
    return adjusted_noise

def mix_audio(gsc_audio, esc_audio, target_snr):
    """Mix GSC audio with ESC audio at target SNR."""
    # Safety check
    if not np.all(np.isfinite(gsc_audio)) or not np.all(np.isfinite(esc_audio)):
        return gsc_audio
        
    # Adjust noise level
    adjusted_noise = adjust_snr(gsc_audio, esc_audio, target_snr)
    
    # Mix audio
    mixed_audio = gsc_audio + adjusted_noise
    
    # Normalize
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio / max_val
        
    return mixed_audio

def process_audio_files(gsc_dir, esc_dir, output_dir, snr_db=10, esc_category=10):
    """Process audio files and save mixed wav files."""
    gsc_dir = Path(gsc_dir)
    esc_dir = Path(esc_dir)
    output_dir = Path(output_dir)
    
    
    # Get ESC audio files
    esc_files_by_category = defaultdict(list)
    for esc_file in esc_dir.glob('*.wav'):
        category = get_esc_category(esc_file)
        esc_files_by_category[category].append(esc_file)
        
    # Get GSC test files
    test_dir = gsc_dir
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
        
    # Process command words
    command_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    cnt = 0
    
    for word in command_words:
        word_dir = test_dir / word
        if not word_dir.exists():
            continue
        # Create output directory
        category_dir = output_dir / f'esc_{esc_category}' / f'{snr_db}db'
        category_dir.mkdir(parents=True, exist_ok=True)
            
        # Get audio files for this word
        audio_files = list(word_dir.glob('*.wav'))
        if not audio_files:
            continue
            
        # Randomly select one audio file
        audio_file = random.choice(audio_files)
                
        # Load GSC audio
        gsc_audio, sr = librosa.load(audio_file, sr=16000)
        
        # Randomly select ESC audio
        esc_files = esc_files_by_category[esc_category]
        esc_file = random.choice(esc_files)
        esc_audio, _ = librosa.load(esc_file, sr=16000)
        
        # Ensure ESC audio is long enough
        if len(esc_audio) < len(gsc_audio):
            esc_audio = np.tile(esc_audio, int(np.ceil(len(gsc_audio) / len(esc_audio))))
        esc_audio = esc_audio[:len(gsc_audio)]
        
        # Mix audio
        mixed_audio = mix_audio(gsc_audio, esc_audio, snr_db)
        
        # Save mixed wav file
        wav_path = category_dir / f'{cnt}.wav'
        sf.write(wav_path, mixed_audio, sr)
        
        # Extract MFCC features
        mfcc_features = extract_mfcc(mixed_audio)
        
        # Save MFCC features
        data = {
            'feature': mfcc_features,
            'label': command_words.index(word)
        }
        np.save(category_dir / f'{cnt}.npy', data)
        
        print(f"Processed {cnt}: {word} mixed with {esc_file.name}")
        cnt += 1

def main():
    gsc_dir = 'speech_commands'
    esc_dir = 'esc50'
    output_dir = 'testwav'
    
    # Process 10 samples with SNR=10dB and ESC category 10
    snr_db = [-10, 0, 10]
    esc_categories = [0, 5, 17, 19, 20, 26, 35, 36, 43, 48]
    for esc_category in esc_categories:
        for snr in snr_db:
            process_audio_files(gsc_dir, esc_dir, output_dir, snr_db=snr, esc_category=esc_category)

if __name__ == "__main__":
    main() 