import yt_dlp as youtube_dl
import os
from vutube_custom import VCtube
from datasets import Dataset, Audio, DatasetDict, load_dataset, concatenate_datasets
from datasets import Dataset, Audio
import pandas as pd
from tqdm import tqdm



def check_korean_subtitles_and_down(video_url, download_path):
    ydl_opts = {
        'quiet': True,
        'subtitleslangs': 'all',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'skip_download': True,
        'format': 'best',
    }
    
    download_path = os.path.join('datasets', download_path, "wavs/" + '%(id)s.%(ext)s')

    # youtube_dl options
    ydl_opts_down = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192'
            }],
            'postprocessors_args': [
                '-ar', '21000'
            ],
            'prefer_ffmpeg': True,
            'keepvideo': False,
            'outtmpl': download_path,  # 다운로드 경로 설정
            'ignoreerrors': True
        }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(video_url, download=False)
            if result:
                subtitles = result.get('subtitles', {})
                # 한국어 자막 또는 자동 생성 자막 확인
                has_korean_subs = 'ko' in subtitles
                if has_korean_subs:
                    try:
                        
                        print('영상 다운로드 {}'.format(video_url))
                        with youtube_dl.YoutubeDL(ydl_opts_down) as ydl_down:
                            ydl_down.download([video_url])
                    except Exception as e:
                        print('error', e)
                
        except Exception as e:
            print(f"에러: {e}")
            return False

def extract_videos_from_playlist(playlist_url):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': True,
        'skip_download': True,
    }
    
    video_list = []
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(playlist_url, download=False)
            if 'entries' in result:
                print("Video links in the playlist:")
                for video in tqdm(result['entries']):
                    video_url = f"https://www.youtube.com/watch?v={video['id']}"
                    video_list.append(video_url)
        except Exception as e:
            print(f"Error occurred: {e}")
    return video_list

def filter_audio_length(example):
    return len(example['name']['array']) > 16000

playlist_name = 'economoy'
playlist_url = 'https://youtube.com/playlist?list=PLh0GSN2S41Yy2-xR3SwdI23x0KvSMcYme&si=ok_9C4ar1Vjv5UPA'
lang = 'ko'
vc = VCtube(playlist_name, playlist_url, lang)

video_list = extract_videos_from_playlist(playlist_url)

for video_url in tqdm(video_list):
    check_korean_subtitles_and_down(video_url, playlist_name)
vc.download_captions()
vc.audio_split()

df = pd.read_csv(f'datasets/{playlist_name}/text/subtitle.csv')
ds = Dataset.from_pandas(df).remove_columns(['Unnamed: 0', 'id', 'start', 'duration']).cast_column("name", Audio(sampling_rate=16000))


ds = ds.filter(filter_audio_length)

train_test_split = ds.train_test_split(test_size=0.2)  # 20% for testing

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

dataset_dict.push_to_hub('Economy')