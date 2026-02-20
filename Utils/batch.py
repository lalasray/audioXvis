from audi_extract import extract_audio_from_video

# Single video
#extract_audio_from_video('path/to/video.mp4', 'output.mp3')

# Batch processing
from audi_extract import batch_extract_audio
batch_extract_audio('/home/lala/Documents/GitHub/audioXvis/data/test_dataset/one_experienced_singer_dataset_songs', '/home/lala/Documents/GitHub/audioXvis/data/test_dataset/one_experienced_singer_dataset_audio/', 'mp3', '/home/lala/miniconda3/envs/audio2vis/bin/ffmpeg')