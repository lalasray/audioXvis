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
import json
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
    
    def get_rms_energy(self, frame_length=2048, hop_length=512):
        """
        Extract RMS (Root Mean Square) energy - measures loudness/amplitude.
        
        Returns:
        --------
        np.ndarray
            RMS energy values
        """
        rms = librosa.feature.rms(y=self.y, frame_length=frame_length,
                                 hop_length=hop_length)[0]
        return rms
    
    def get_pitch(self, fmin=80, fmax=400, hop_length=512):
        """
        Extract fundamental frequency (pitch/F0) using YIN algorithm.
        
        Parameters:
        -----------
        fmin : int
            Minimum frequency to consider (Hz)
        fmax : int
            Maximum frequency to consider (Hz)
        hop_length : int
            Number of samples between frames
            
        Returns:
        --------
        np.ndarray
            Fundamental frequency values (0 if unvoiced)
        """
        f0 = librosa.yin(self.y, fmin=fmin, fmax=fmax, hop_length=hop_length)
        return f0
    
    def get_onset_strength(self, hop_length=512):
        """
        Extract onset strength - detects attack points in the signal.
        
        Returns:
        --------
        np.ndarray
            Onset strength values
        """
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr,
                                                hop_length=hop_length)
        return onset_env
    
    def get_spectral_contrast(self, n_fft=2048, hop_length=512):
        """
        Extract spectral contrast - distinguishes peaks vs valleys in spectrum.
        
        Returns:
        --------
        np.ndarray
            Spectral contrast values (one value per band per frame)
        """
        contrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr,
                                                     n_fft=n_fft, hop_length=hop_length)
        return contrast
    
    def get_harmonic_percussive(self, margin=2.0):
        """
        Separate harmonic and percussive components from audio.
        
        Parameters:
        -----------
        margin : float
            Margin for HPSS algorithm
            
        Returns:
        --------
        tuple
            (harmonic_component, percussive_component)
        """
        harmonic, percussive = librosa.effects.hpss(self.y, margin=margin)
        return harmonic, percussive
    
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
    
    def plot_tempogram(self, figsize=(14, 8), output_path=None):
        """
        Plot tempogram (tempo/rhythm information).
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        tempogram = self.get_tempogram()
        
        plt.figure(figsize=figsize)
        img = librosa.display.specshow(tempogram, sr=self.sr, x_axis='time',
                                       y_axis='tempo', cmap='viridis')
        plt.colorbar(img, label='Strength')
        plt.title('Tempogram (Rhythm/Tempo)')
        plt.xlabel('Time (s)')
        plt.ylabel('Tempo (BPM)')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved tempogram to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_rms_energy(self, figsize=(14, 8), output_path=None):
        """
        Plot RMS energy (loudness/amplitude envelope).
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        hop_length = 512
        rms = self.get_rms_energy(hop_length=hop_length)
        S_db = self.get_mel_spectrogram()
        
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time',
                                       y_axis='mel', cmap='magma', ax=ax)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Plot RMS energy
        ax.plot(librosa.frames_to_time(np.arange(len(rms)), sr=self.sr),
               rms, label='RMS Energy', color='red', linewidth=2)
        
        ax.set_title('Mel-Spectrogram with RMS Energy Envelope')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved RMS energy to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_pitch(self, figsize=(14, 6), output_path=None):
        """
        Plot fundamental frequency/pitch contour.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        f0 = self.get_pitch()
        
        plt.figure(figsize=figsize)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sr, hop_length=512)
        
        # Plot pitch (mask out 0 values which indicate unvoiced)
        voiced = f0 > 0
        plt.scatter(times[voiced], f0[voiced], alpha=0.6, s=10, color='blue', label='Voiced Pitch')
        plt.plot(times[voiced], f0[voiced], alpha=0.3, color='blue', linewidth=1)
        
        plt.title('Fundamental Frequency (Pitch/F0) Contour')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True, alpha=0.3)
        plt.ylim([50, 450])
        plt.legend()
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved pitch contour to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_onset_strength(self, figsize=(14, 8), output_path=None):
        """
        Plot onset strength (attack/pronunciation detection).
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        hop_length = 512
        onset = self.get_onset_strength(hop_length=hop_length)
        S_db = self.get_mel_spectrogram()
        
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time',
                                       y_axis='mel', cmap='magma', ax=ax)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Plot onset strength
        ax2 = ax.twinx()
        ax2.plot(librosa.frames_to_time(np.arange(len(onset)), sr=self.sr),
                onset, label='Onset Strength', color='cyan', linewidth=2)
        ax2.set_ylabel('Onset Strength')
        
        ax.set_title('Mel-Spectrogram with Onset Strength')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved onset strength to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_spectral_contrast(self, figsize=(14, 8), output_path=None):
        """
        Plot spectral contrast (peaks vs valleys in spectrum).
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        contrast = self.get_spectral_contrast()
        
        plt.figure(figsize=figsize)
        img = librosa.display.specshow(contrast, sr=self.sr, x_axis='time',
                                       y_axis='linear', cmap='viridis')
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Spectral Contrast (Peaks vs Valleys)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency Band')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved spectral contrast to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_harmonic_percussive(self, figsize=(14, 10), output_path=None):
        """
        Plot harmonic and percussive separation.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the figure
        """
        harmonic, percussive = self.get_harmonic_percussive()
        
        # Compute spectrograms for visualization
        S_harmonic = np.abs(librosa.stft(harmonic))
        S_percussive = np.abs(librosa.stft(percussive))
        S_h_db = librosa.amplitude_to_db(S_harmonic, ref=np.max)
        S_p_db = librosa.amplitude_to_db(S_percussive, ref=np.max)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Harmonic
        img1 = librosa.display.specshow(S_h_db, sr=self.sr, x_axis='time',
                                        y_axis='log', cmap='magma', ax=axes[0])
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
        axes[0].set_title('Harmonic Component (Vocal Harmonics)')
        axes[0].set_ylabel('Frequency (Hz)')
        
        # Percussive
        img2 = librosa.display.specshow(S_p_db, sr=self.sr, x_axis='time',
                                        y_axis='log', cmap='magma', ax=axes[1])
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
        axes[1].set_title('Percussive Component (Noise/Consonants)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved harmonic/percussive separation to: {output_path}")
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
            ('tempogram', self.plot_tempogram, {}),
            ('rms_energy', self.plot_rms_energy, {}),
            ('pitch', self.plot_pitch, {}),
            ('onset_strength', self.plot_onset_strength, {}),
            ('spectral_contrast', self.plot_spectral_contrast, {}),
            ('harmonic_percussive', self.plot_harmonic_percussive, {}),
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
            'rms_energy': self.get_rms_energy(),
            'pitch': self.get_pitch(),
            'onset_strength': self.get_onset_strength(),
            'spectral_contrast': self.get_spectral_contrast(),
            'harmonic_percussive': self.get_harmonic_percussive(),
        }
        return features

    @staticmethod
    def _safe_zscore(x, axis=None, eps=1e-8):
        """Z-score normalize with numerical safety."""
        x = np.asarray(x, dtype=np.float32)
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        return (x - mean) / std

    @staticmethod
    def _safe_minmax(x, axis=None, eps=1e-8):
        """Min-max normalize to [0, 1] with numerical safety."""
        x = np.asarray(x, dtype=np.float32)
        x_min = np.min(x, axis=axis, keepdims=True)
        x_max = np.max(x, axis=axis, keepdims=True)
        denom = np.where((x_max - x_min) < eps, 1.0, (x_max - x_min))
        return (x - x_min) / denom

    def normalize_features(self, features):
        """
        Normalize extracted features for improved cross-user comparability.

        Returns:
        --------
        tuple[dict, dict]
            (normalized_features, normalization_metadata)
        """
        normalized = {}
        normalization_meta = {}

        zscore_keys = {
            'mel_spectrogram',
            'spectrogram',
            'mfcc',
            'chroma',
            'tempogram',
            'spectral_contrast',
        }

        minmax_keys = {
            'waveform',
            'rms_energy',
            'spectral_centroid',
            'spectral_rolloff',
            'zero_crossing_rate',
            'onset_strength',
        }

        for key, value in features.items():
            if key == 'harmonic_percussive' and isinstance(value, tuple) and len(value) == 2:
                harmonic, percussive = value
                harmonic = np.asarray(harmonic, dtype=np.float32)
                percussive = np.asarray(percussive, dtype=np.float32)

                harmonic_norm = self._safe_minmax(harmonic)
                percussive_norm = self._safe_minmax(percussive)

                total_energy = np.abs(harmonic) + np.abs(percussive)
                total_energy = np.where(total_energy < 1e-8, 1.0, total_energy)
                harmonic_ratio = np.abs(harmonic) / total_energy

                normalized['harmonic_component'] = harmonic_norm
                normalized['percussive_component'] = percussive_norm
                normalized['harmonic_ratio'] = harmonic_ratio.astype(np.float32)

                normalization_meta['harmonic_component'] = 'minmax'
                normalization_meta['percussive_component'] = 'minmax'
                normalization_meta['harmonic_ratio'] = 'ratio_abs_h_over_abs_h_plus_abs_p'
                continue

            arr = np.asarray(value, dtype=np.float32)

            if key == 'pitch':
                voiced_mask = arr > 0
                out = np.zeros_like(arr, dtype=np.float32)
                if np.any(voiced_mask):
                    voiced_values = arr[voiced_mask]
                    voiced_min = np.min(voiced_values)
                    voiced_max = np.max(voiced_values)
                    denom = (voiced_max - voiced_min) if (voiced_max - voiced_min) > 1e-8 else 1.0
                    out[voiced_mask] = (voiced_values - voiced_min) / denom
                normalized[key] = out
                normalization_meta[key] = 'voiced_only_minmax_unvoiced_zero'
                continue

            if key in zscore_keys:
                if arr.ndim >= 2:
                    normalized[key] = self._safe_zscore(arr, axis=-1)
                    normalization_meta[key] = 'zscore_per_band_over_time'
                else:
                    normalized[key] = self._safe_zscore(arr)
                    normalization_meta[key] = 'zscore_global'
            elif key in minmax_keys:
                normalized[key] = self._safe_minmax(arr)
                normalization_meta[key] = 'minmax_global'
            else:
                normalized[key] = arr
                normalization_meta[key] = 'none'

        return normalized, normalization_meta

    def save_all_features(self, output_path):
        """
        Extract all features, normalize them, and save to a compressed .npz file.

        Parameters:
        -----------
        output_path : str
            Path to save feature archive (.npz)
        """
        features = self.extract_all_features()
        features, normalization_meta = self.normalize_features(features)

        serializable = {}
        summary = {}
        summary['normalization'] = normalization_meta
        for key, value in features.items():
            serializable[key] = value
            if hasattr(value, 'shape'):
                summary[f'{key}_shape'] = list(value.shape)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **serializable)

        summary_path = output_path.with_suffix('.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved features to: {output_path}")
        print(f"Saved feature summary to: {summary_path}")


def process_audio_directory(audio_directory, output_directory=None, plot_all=True,
                            save_features=False, features_directory=None):
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
    save_features : bool, optional
        Whether to save extracted feature arrays for each file
    features_directory : str, optional
        Directory to save feature .npz/.json files
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

    if output_directory:
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    if save_features:
        if features_directory is None:
            if output_directory:
                features_directory = str(Path(output_directory) / 'features')
            else:
                features_directory = str(audio_dir / 'audio_features')
        Path(features_directory).mkdir(parents=True, exist_ok=True)
    
    for audio_file in audio_files:
        try:
            print(f"Processing: {audio_file.name}")
            extractor = AudioFeatureExtractor(str(audio_file))

            if save_features:
                features_out = Path(features_directory) / f"{audio_file.stem}_features.npz"
                extractor.save_all_features(str(features_out))
            
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
        print("Usage: python audio_feature_extraction.py <audio_file_path> [--plot-all] [--output-dir OUTPUT_DIR] [--save-features] [--features-dir FEATURES_DIR]")
        print("\nExamples:")
        print("  python audio_feature_extraction.py audio.mp3")
        print("  python audio_feature_extraction.py audio.mp3 --plot-all")
        print("  python audio_feature_extraction.py audio.mp3 --output-dir ./plots/")
        print("  python audio_feature_extraction.py audio.mp3 --output-dir ./plots/ --save-features --features-dir ./features/")
        print("\nTo process a directory:")
        print("  python audio_feature_extraction.py --directory audio_folder/ --output-dir ./plots/ --save-features")
    else:
        if sys.argv[1] == "--directory":
            # Directory mode
            if len(sys.argv) < 3:
                print("Error: Please provide directory path")
                sys.exit(1)
            
            audio_dir = sys.argv[2]
            output_dir = None
            save_features = "--save-features" in sys.argv
            features_dir = None
            
            if "--output-dir" in sys.argv:
                output_dir = sys.argv[sys.argv.index("--output-dir") + 1]

            if "--features-dir" in sys.argv:
                features_dir = sys.argv[sys.argv.index("--features-dir") + 1]
            
            process_audio_directory(audio_dir, output_dir, plot_all=True,
                                    save_features=save_features,
                                    features_directory=features_dir)
        else:
            # Single file mode
            audio_file = sys.argv[1]
            plot_all = "--plot-all" in sys.argv
            output_dir = None
            save_features = "--save-features" in sys.argv
            features_dir = None
            
            if "--output-dir" in sys.argv:
                output_dir = sys.argv[sys.argv.index("--output-dir") + 1]

            if "--features-dir" in sys.argv:
                features_dir = sys.argv[sys.argv.index("--features-dir") + 1]
            
            extractor = AudioFeatureExtractor(audio_file)

            if save_features:
                if features_dir is None:
                    if output_dir:
                        features_dir = str(Path(output_dir) / 'features')
                    else:
                        features_dir = str(Path(audio_file).parent / 'audio_features')
                Path(features_dir).mkdir(parents=True, exist_ok=True)
                feature_path = Path(features_dir) / f"{Path(audio_file).stem}_features.npz"
                extractor.save_all_features(str(feature_path))
            
            if plot_all:
                extractor.plot_all_features(output_dir)
            else:
                # Default: plot mel-spectrogram
                output_path = None
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_path = str(Path(output_dir) / f"{Path(audio_file).stem}_mel_spectrogram.png")
                
                extractor.plot_mel_spectrogram(output_path=output_path)
