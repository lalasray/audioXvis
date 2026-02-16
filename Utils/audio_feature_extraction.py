"""
Audio feature extraction and visualization module.
Extracts various audio features and provides multiple visualization forms.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from librosa import feature
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract and visualize various audio features from audio files."""
    
    def __init__(self, audio_path, sr=22050):
        """
        Initialize the audio feature extractor.
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        sr : int, optional
            Sample rate (default: 22050 Hz)
        """
        self.audio_path = audio_path
        self.sr = sr
        self.y, self.sr = librosa.load(audio_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.time_axis = librosa.frames_to_time(np.arange(len(self.y)), sr=self.sr)
        
        print(f"Loaded audio: {audio_path}")
        print(f"Duration: {self.duration:.2f}s, Sample Rate: {self.sr} Hz")
    
    def get_mel_spectrogram(self, n_mels=128, n_fft=2048, hop_length=512):
        """
        Extract mel-spectrogram from audio.
        
        Parameters:
        -----------
        n_mels : int
            Number of mel bands
        n_fft : int
            FFT window size
        hop_length : int
            Number of samples between frames
            
        Returns:
        --------
        np.ndarray
            Mel-spectrogram in dB scale
        """
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=n_mels, 
                                          n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    
    def get_spectrogram(self, n_fft=2048, hop_length=512):
        """
        Extract short-time Fourier transform (STFT) spectrogram.
        
        Returns:
        --------
        np.ndarray
            Spectrogram in dB scale
        """
        D = librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        return S_db
    
    def get_mfcc(self, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCC).
        
        Parameters:
        -----------
        n_mfcc : int
            Number of MFCCs to extract
            
        Returns:
        --------
        np.ndarray
            MFCC features
        """
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_mfcc,
                                    n_fft=n_fft, hop_length=hop_length)
        return mfcc
    
    def get_chroma(self, n_fft=2048, hop_length=512):
        """
        Extract chroma features (pitch class profile).
        
        Returns:
        --------
        np.ndarray
            Chroma features (12D vector for each time frame)
        """
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr,
                                            n_fft=n_fft, hop_length=hop_length)
        return chroma
    
    def get_tempogram(self, hop_length=512):
        """
        Extract tempogram (tempo information).
        
        Returns:
        --------
        np.ndarray
            Tempogram
        """
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(y=self.y, sr=self.sr, hop_length=hop_length)
        return tempogram
    
    def get_spectral_centroid(self, n_fft=2048, hop_length=512):
        """
        Extract spectral centroid (center of mass of spectrum).
        
        Returns:
        --------
        np.ndarray
            Spectral centroid values
        """
        cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr,
                                                n_fft=n_fft, hop_length=hop_length)[0]
        return cent
    
    def get_spectral_rolloff(self, n_fft=2048, hop_length=512):
        """
        Extract spectral rolloff frequency.
        
        Returns:
        --------
        np.ndarray
            Spectral rolloff values
        """
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr,
                                                  n_fft=n_fft, hop_length=hop_length)[0]
        return rolloff
    
    def get_zero_crossing_rate(self, frame_length=2048, hop_length=512):
        """
        Extract zero-crossing rate (measure of noisiness).
        
        Returns:
        --------
        np.ndarray
            Zero-crossing rate values
        """
        zcr = librosa.feature.zero_crossing_rate(self.y, frame_length=frame_length,
                                                hop_length=hop_length)[0]
        return zcr
    
    def plot_waveform(self, figsize=(14, 4), output_path=None):
        """
        Plot raw audio waveform.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        plt.figure(figsize=figsize)
        librosa.display.waveshow(self.y, sr=self.sr)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved waveform to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_mel_spectrogram(self, n_mels=128, figsize=(14, 8), output_path=None):
        """
        Plot mel-spectrogram.
        
        Parameters:
        -----------
        n_mels : int
            Number of mel bands
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        S_db = self.get_mel_spectrogram(n_mels=n_mels)
        
        plt.figure(figsize=figsize)
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time',
                                       y_axis='mel', cmap='magma')
        plt.colorbar(img, format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram ({n_mels} bands)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved mel-spectrogram to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_spectrogram(self, figsize=(14, 8), output_path=None):
        """
        Plot STFT spectrogram.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        S_db = self.get_spectrogram()
        
        plt.figure(figsize=figsize)
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time',
                                       y_axis='linear', cmap='magma')
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Spectrogram (Linear Scale)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved spectrogram to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_mfcc(self, n_mfcc=13, figsize=(14, 8), output_path=None):
        """
        Plot MFCC features.
        
        Parameters:
        -----------
        n_mfcc : int
            Number of MFCCs
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        mfcc = self.get_mfcc(n_mfcc=n_mfcc)
        
        plt.figure(figsize=figsize)
        img = librosa.display.specshow(mfcc, sr=self.sr, x_axis='time',
                                       y_axis='linear', cmap='viridis')
        plt.colorbar(img)
        plt.title(f'MFCC ({n_mfcc} coefficients)')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficient')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved MFCC to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_chroma(self, figsize=(14, 6), output_path=None):
        """
        Plot chroma features.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        chroma = self.get_chroma()
        
        plt.figure(figsize=figsize)
        img = librosa.display.specshow(chroma, sr=self.sr, x_axis='time',
                                       y_axis='chroma', cmap='hsv')
        plt.colorbar(img)
        plt.title('Chromagram (Pitch Class Profile)')
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Class')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved chromagram to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_spectral_features(self, figsize=(14, 8), output_path=None):
        """
        Plot spectral features (centroid, rolloff, ZCR).
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        # Get time frames for spectral features
        hop_length = 512
        frames = np.arange(len(self.y)) // hop_length
        t_frames = librosa.frames_to_time(np.arange(len(self.y)) // hop_length, sr=self.sr)
        
        cent = self.get_spectral_centroid(hop_length=hop_length)
        rolloff = self.get_spectral_rolloff(hop_length=hop_length)
        zcr = self.get_zero_crossing_rate(hop_length=hop_length)
        
        # Normalize for visualization
        S_db = self.get_mel_spectrogram()
        
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time',
                                       y_axis='mel', cmap='magma', ax=ax)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Plot spectral features on top
        ax.plot(librosa.frames_to_time(np.arange(len(cent)), sr=self.sr),
                cent, label='Spectral Centroid', color='cyan', linewidth=2)
        ax.plot(librosa.frames_to_time(np.arange(len(rolloff)), sr=self.sr),
                rolloff, label='Spectral Rolloff', color='yellow', linewidth=2)
        
        # Plot ZCR on secondary axis
        ax2 = ax.twinx()
        ax2.plot(librosa.frames_to_time(np.arange(len(zcr)), sr=self.sr),
                zcr, label='Zero Crossing Rate', color='lime', linewidth=2, linestyle='--')
        ax2.set_ylabel('Zero Crossing Rate')
        
        ax.set_title('Mel-Spectrogram with Spectral Features')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved spectral features plot to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_all_features(self, output_dir=None):
        """
        Plot all available features in a single visualization.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save individual plots
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        features_to_plot = [
            ('waveform', self.plot_waveform, {}),
            ('spectrogram', self.plot_spectrogram, {}),
            ('mel_spectrogram', self.plot_mel_spectrogram, {}),
            ('mfcc', self.plot_mfcc, {}),
            ('chroma', self.plot_chroma, {}),
            ('spectral_features', self.plot_spectral_features, {}),
        ]
        
        for name, plot_func, kwargs in features_to_plot:
            output_path = None
            if output_dir:
                output_path = str(output_dir / f"{Path(self.audio_path).stem}_{name}.png")
            
            print(f"Generating {name}...")
            try:
                plot_func(output_path=output_path, **kwargs)
            except Exception as e:
                print(f"Error generating {name}: {str(e)}")
                continue
    
    def extract_all_features(self):
        """
        Extract all features and return as dictionary.
        
        Returns:
        --------
        dict
            Dictionary containing all extracted features
        """
        features = {
            'waveform': self.y,
            'mel_spectrogram': self.get_mel_spectrogram(),
            'spectrogram': self.get_spectrogram(),
            'mfcc': self.get_mfcc(),
            'chroma': self.get_chroma(),
            'spectral_centroid': self.get_spectral_centroid(),
            'spectral_rolloff': self.get_spectral_rolloff(),
            'zero_crossing_rate': self.get_zero_crossing_rate(),
            'tempogram': self.get_tempogram(),
        }
        return features


def process_audio_directory(audio_directory, output_directory=None, plot_all=True):
    """
    Process all audio files in a directory.
    
    Parameters:
    -----------
    audio_directory : str
        Directory containing audio files
    output_directory : str, optional
        Directory to save visualizations
    plot_all : bool, optional
        Whether to plot all features for each file
    """
    # Use non-interactive backend for batch processing
    import matplotlib
    matplotlib.use('Agg')
    
    audio_dir = Path(audio_directory)
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.m4a']
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f'*{ext}'))
        audio_files.extend(audio_dir.glob(f'*{ext.upper()}'))
    
    if not audio_files:
        print(f"No audio files found in {audio_directory}")
        return
    
    print(f"Found {len(audio_files)} audio files\n")
    
    for audio_file in audio_files:
        try:
            print(f"Processing: {audio_file.name}")
            extractor = AudioFeatureExtractor(str(audio_file))
            
            if plot_all and output_directory:
                extractor.plot_all_features(output_directory)
            elif output_directory:
                # Just save mel-spectrogram by default
                output_path = Path(output_directory) / f"{audio_file.stem}_mel_spectrogram.png"
                extractor.plot_mel_spectrogram(output_path=str(output_path))
        
        except Exception as e:
            print(f"Error processing {audio_file.name}: {str(e)}")
            continue


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_feature_extraction.py <audio_file_path> [--plot-all] [--output-dir OUTPUT_DIR]")
        print("\nExamples:")
        print("  python audio_feature_extraction.py audio.mp3")
        print("  python audio_feature_extraction.py audio.mp3 --plot-all")
        print("  python audio_feature_extraction.py audio.mp3 --output-dir ./plots/")
        print("\nTo process a directory:")
        print("  python audio_feature_extraction.py --directory audio_folder/ --output-dir ./plots/")
    else:
        if sys.argv[1] == "--directory":
            # Directory mode
            if len(sys.argv) < 3:
                print("Error: Please provide directory path")
                sys.exit(1)
            
            audio_dir = sys.argv[2]
            output_dir = None
            
            if "--output-dir" in sys.argv:
                output_dir = sys.argv[sys.argv.index("--output-dir") + 1]
            
            process_audio_directory(audio_dir, output_dir)
        else:
            # Single file mode
            audio_file = sys.argv[1]
            plot_all = "--plot-all" in sys.argv
            output_dir = None
            
            if "--output-dir" in sys.argv:
                output_dir = sys.argv[sys.argv.index("--output-dir") + 1]
            
            extractor = AudioFeatureExtractor(audio_file)
            
            if plot_all:
                extractor.plot_all_features(output_dir)
            else:
                # Default: plot mel-spectrogram
                output_path = None
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_path = str(Path(output_dir) / f"{Path(audio_file).stem}_mel_spectrogram.png")
                
                extractor.plot_mel_spectrogram(output_path=output_path)
