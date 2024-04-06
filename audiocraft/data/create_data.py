from pytube import YouTube
from moviepy.editor import *
import os
import shutil
import random

import json
from pathlib import Path

from audiocraft.data.audio import audio_read
from audiocraft.data.essentia_utils import get_essentia_features
from audiocraft.data.description_generator import DescriptionGenerator

import numpy as np
import soundfile as sf

from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

from pytube.innertube import _default_clients 
_default_clients["ANDROID_MUSIC"] = _default_clients["IOS"]

import warnings
warnings.filterwarnings("ignore")



def download_audio(url, data_dict, save_path='../Dataset/raw_music/general/', overwrite=False):
    yt = YouTube(
        url,
        use_oauth=False,
        allow_oauth_cache=False
    )

    assert os.path.exists(save_path), "Save path doesn't exist!"
    
    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    
    filename = audio_stream.default_filename
    download_path = Path(save_path) / filename

    wav_filename_path = download_path.with_suffix('.wav')
    json_filename_path = download_path.with_suffix('.json')

    if wav_filename_path.exists() and json_filename_path.exists() and not overwrite:
        print(f'{wav_filename_path} \033[1malready exists. Skipping...\033[0m')
        return

    audio_stream.download(output_path=save_path, filename=filename)

    clip = AudioFileClip(str(download_path))
    wav_filename = str(wav_filename_path)
    clip.write_audiofile(wav_filename)
    
    with open(json_filename_path, 'w') as file:
        json.dump(data_dict, file)

    os.remove(download_path)


def download_split(split='train', link_core_path='../Dataset/youtube_music_links/', save_core_path = '../Dataset/raw_music/', overwrite=False):
    link_folder = Path(link_core_path) / split
    link_file = link_folder / 'links.jsonl'
    assert link_file.exists(), f"There is no {link_file}."

    save_path = Path(save_core_path) / split
    if not save_path.exists():
        os.makedirs(save_path)
        
    with open(link_file, "r") as text_file:
        for line in text_file:
            data_dict = json.loads(line)
            link = data_dict.pop('link')
            download_audio(link, data_dict, save_path, overwrite=overwrite)


def divide_into_clips(split='train', raw_music_path='../Dataset/raw_music/', clip_duration=30, stride=15):
    full_path = Path(raw_music_path) / split
    wav_files = full_path.glob('*.wav')
    
    for wav_file in wav_files:
        json_file_path = wav_file.with_suffix('.json')
        audio, sr = audio_read(wav_file)
        
        # Make mono
        audio = audio.mean(0)
        samples_per_clip = clip_duration * sr
        step_size = stride * sr
        total_clips = int(np.ceil((len(audio) - samples_per_clip) / step_size)) + 1
        for i in range(total_clips):
            start_sample = i * step_size
            end_sample = start_sample + samples_per_clip
            clip = audio[start_sample:end_sample]
            
            clip_name = wav_file.parent / f"{wav_file.stem}_{i+1}"
            new_clip_path = clip_name.with_suffix('.wav')
            new_json_path = clip_name.with_suffix('.json')
            shutil.copy(json_file_path, new_json_path)
            sf.write(new_clip_path, clip, sr)
        os.remove(wav_file)
        os.remove(json_file_path)
        print('Done:', wav_file)


def prepare_attributes(split='train', core_music_folder='../Dataset/raw_music/'):
    music_folder = Path(core_music_folder) / split
    music_files = music_folder.glob('*.wav')
    
    for music_path in music_files:
        music_file_name = music_path.stem
        json_file_path = music_path.with_suffix('.json')

        with open(json_file_path, "r") as json_file:
            data_dict = json.load(json_file)
            
        attribute_dict = {}
        attribute_dict["key"] = ""
        attribute_dict["artist"] = data_dict["artist"]
        attribute_dict["sample_rate"] = 0
        attribute_dict["file_extension"] = music_path.suffix[1:]
        attribute_dict["description"] = ""
        attribute_dict["keywords"] = ""
        attribute_dict["duration"] = 0
        attribute_dict["bpm"] = ""
        attribute_dict["genre"] = ""
        attribute_dict["title"] = ""
        attribute_dict["name"] = music_file_name
        attribute_dict["instrument"] = ""
        attribute_dict["moods"] = []
        attribute_dict["label"] = data_dict["label"]

        with open(json_file_path, "w") as json_file:
            json.dump(attribute_dict, json_file)
            
        print('\033[1mSaved Json file for:\033[0m', music_file_name)


def fill_json_split(split='train', main_files_path='../Dataset/raw_music/', n_best=3, jsonl_path='egs/', weights_path='../Dataset/essentia_weights/'):
    jsonl_main_path = Path(jsonl_path) / split
    jsonl_full_path = jsonl_main_path / 'data.jsonl'
    
    full_path = Path(main_files_path) / split 
    all_files = full_path.glob('*.wav')
    
    if not jsonl_main_path.exists():
        os.makedirs(jsonl_main_path)
        
    with open(jsonl_full_path, 'w') as jsonl_file:
        for file in all_files:
            jsonl_info = fill_json(file, n_best=n_best, weights_path=weights_path)
            json_str = json.dumps(jsonl_info)
            jsonl_file.write(json_str + '\n')


def fill_json(music_file, n_best=3, weights_path = '../Dataset/essentia_weights/'):
    str_path_music = str(music_file)
    json_file_path = music_file.with_suffix('.json')
    
    audio, sr = audio_read(music_file)
    audio = audio[0]
    sec_len = len(audio)/sr

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    labeling_type = data['label']

    if labeling_type == 'essentia':
        music_info = get_essentia_features(audio_filename=str_path_music, n_best = n_best, weights_folder=weights_path)
    else:
        music_info = custom_labeler(labeling_type, n_best=n_best)
        
    data['duration'] = sec_len
    data['sample_rate'] = sr
    data['genre'] = music_info['genres']
    data['instrument'] = ', '.join(music_info['instruments'])
    data['moods'] = music_info['moods'].split(', ')

    with open(json_file_path, "w") as json_file:
            json.dump(data, json_file)

    jsonl_data = {}

    jsonl_data['path'] = str_path_music
    jsonl_data['duration'] = sec_len
    jsonl_data['sample_rate'] = sr
    jsonl_data['amplitude'] = None
    jsonl_data['weight'] = None
    jsonl_data['info_path'] = None

    return jsonl_data

    print('Done:', music_file)

def fill_descriptions(split='train', main_path='../Dataset/raw_music/', llm_checkpoint = 'ericzzz/falcon-rw-1b-instruct-openorca'):
    full_path = Path(main_path) / split 
    json_files = full_path.glob('*.json')
    description_generator = DescriptionGenerator(checkpoint = llm_checkpoint)

    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
        description = description_generator.generate(data)
        data['description'] = description
        with open(json_file, "w") as file:
            json.dump(data, file)
        print('\nDone:', json_file)
        print('Description:', description)
        print('-'*50)


def custom_labeler(label, n_best = 3):
    result_dict = {} 
    if label=='duduk':
        result_dict['genres'] = 'Armenian, Armenian Folk'
        result_dict['instruments'] = ['duduk']   
        result_dict['moods'] = ', '.join(random.sample(['melancholic', 'emotional', 'dramatic', 'relaxing', 'sad'], n_best))
    return result_dict