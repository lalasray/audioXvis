"""
Audio extraction utility for extracting audio from video files.
"""

import os
import subprocess
from pathlib import Path


def extract_audio_from_video(video_path, output_path=None, audio_format='mp3', ffmpeg_path=None):
    """
    Extract audio from a video file and save it using ffmpeg.
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file
    output_path : str, optional
        Path where the audio file will be saved.
        If None, saves in the same directory with the video name
    audio_format : str, optional
        Audio format to save as ('mp3', 'wav', 'aac', 'flac')
        Default is 'mp3'
    ffmpeg_path : str, optional
        Full path to ffmpeg executable. If None, searches in PATH
    
    Returns:
    --------
    str
        Path to the saved audio file
    
    Raises:
    -------
    FileNotFoundError
        If the video file doesn't exist
    ValueError
        If the audio format is not supported
    """
    
    # Validate video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Validate audio format
    supported_formats = ['mp3', 'wav', 'aac', 'flac']
    if audio_format.lower() not in supported_formats:
        raise ValueError(f"Unsupported audio format. Supported: {supported_formats}")
    
    try:
        # Find ffmpeg if not provided
        if ffmpeg_path is None:
            # Try to find ffmpeg in common locations
            possible_paths = [
                'ffmpeg',  # In PATH
                '/usr/bin/ffmpeg',
                '/usr/local/bin/ffmpeg',
                os.path.expanduser('~/miniconda3/bin/ffmpeg'),
                os.path.expanduser('~/miniconda3/envs/audio2vis/bin/ffmpeg'),
                '/home/lala/miniconda3/envs/audio2vis/bin/ffmpeg'
            ]
            
            for path in possible_paths:
                if os.path.exists(path) or path == 'ffmpeg':
                    ffmpeg_path = path
                    break
            
            if ffmpeg_path is None:
                raise FileNotFoundError("FFmpeg not found. Please install ffmpeg or provide ffmpeg_path parameter")
        
        # Determine output path
        if output_path is None:
            video_name = Path(video_path).stem
            video_dir = Path(video_path).parent
            output_path = str(video_dir / f"{video_name}.{audio_format}")
        else:
            # Ensure output_path has the correct extension
            if not output_path.endswith(f".{audio_format}"):
                output_path = f"{output_path}.{audio_format}"
        
        # Extract and save audio using ffmpeg
        print(f"Extracting audio from: {video_path}")
        
        # FFmpeg command to extract audio
        audio_codec_map = {
            'mp3': 'libmp3lame',
            'wav': 'pcm_s16le',
            'aac': 'aac',
            'flac': 'flac'
        }
        
        audio_codec = audio_codec_map[audio_format.lower()]
        
        cmd = [
            ffmpeg_path,
            '-i', video_path,
            '-vn',  # No video
            '-acodec', audio_codec,
            '-ab', '192k',  # Audio bitrate
            '-y',  # Overwrite output file
            output_path
        ]
        
        # Run ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        
        print(f"Audio successfully saved to: {output_path}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        raise


def batch_extract_audio(video_directory, output_directory=None, audio_format='mp3', ffmpeg_path=None):
    """
    Extract audio from all video files in a directory.
    
    Parameters:
    -----------
    video_directory : str
        Directory containing video files
    output_directory : str, optional
        Directory to save audio files. If None, saves in the same directory as videos
    audio_format : str, optional
        Audio format to save as ('mp3', 'wav', 'aac', 'flac')
    ffmpeg_path : str, optional
        Full path to ffmpeg executable. If None, searches in PATH
    
    Returns:
    --------
    list
        List of paths to extracted audio files
    """
    
    # Video file extensions to look for
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_directory).glob(f'*{ext}'))
        video_files.extend(Path(video_directory).glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"No video files found in {video_directory}")
        return []
    
    # Create output directory if specified and doesn't exist
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    extracted_audio_files = []
    
    for video_file in video_files:
        try:
            if output_directory:
                audio_name = Path(video_file).stem
                output_path = os.path.join(output_directory, f"{audio_name}.{audio_format}")
            else:
                output_path = None
            
            audio_path = extract_audio_from_video(str(video_file), output_path, audio_format, ffmpeg_path)
            extracted_audio_files.append(audio_path)
        
        except Exception as e:
            print(f"Failed to extract audio from {video_file}: {str(e)}")
            continue
    
    return extracted_audio_files


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audi_extract.py <video_file_path> [output_path] [format]")
        print("Supported formats: mp3, wav, aac, flac")
        print("\nExample:")
        print("  python audi_extract.py video.mp4")
        print("  python audi_extract.py video.mp4 audio.wav wav")
    else:
        video_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        format_type = sys.argv[3] if len(sys.argv) > 3 else 'mp3'
        
        extract_audio_from_video(video_file, output_file, format_type)
