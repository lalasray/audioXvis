from audi_extract import extract_audio_from_video

# Single video
#extract_audio_from_video('path/to/video.mp4', 'output.mp3')

# Batch processing
from audi_extract import batch_extract_audio
batch_extract_audio('/home/lala/Documents/GitHub/audioXvis/data/dataset/', '/home/lala/Documents/GitHub/audioXvis/data/dataset/audio/', 'mp3', '/home/lala/miniconda3/envs/audio2vis/bin/ffmpeg')