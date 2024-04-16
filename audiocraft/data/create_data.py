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



def download_audio(url, link_info_dict, save_path='../Dataset/raw_music/general/', overwrite=False):
    yt = YouTube(
        url,
        use_oauth=False,
        allow_oauth_cache=False
    )

    assert save_path.exists(), "Save path doesn't exist!"
    
    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    
    audio_filename = audio_stream.default_filename
    download_path = save_path / audio_filename

    wav_save_path = download_path.with_suffix('.wav')
    json_save_path = download_path.with_suffix('.json')

    if wav_save_path.exists() and json_save_path.exists() and not overwrite:
        print(f'{wav_save_path} \033[1malready exists. Skipping...\033[0m')
        return

    audio_stream.download(output_path=save_path, filename=audio_filename)

    audio_clip = AudioFileClip(str(download_path))
    wav_filename = str(wav_save_path)
    audio_clip.write_audiofile(wav_filename)
    
    with open(json_save_path, 'w') as json_file:
        json.dump(link_info_dict, json_file)

    os.remove(download_path)


def download_split(split='train', link_core_path='../Dataset/youtube_music_links/', save_core_path = '../Dataset/raw_music/', overwrite=False):
    link_folder = Path(link_core_path) / split
    link_file = link_folder / 'links.jsonl'
    assert link_file.exists(), f"There is no {link_file}."

    save_path = Path(save_core_path) / split
    if not save_path.exists():
        os.makedirs(save_path)
        
    with open(link_file, "r") as opened_link_file:
        for line in opened_link_file:
            link_info_dict = json.loads(line)
            youtube_link = link_info_dict.pop('link')
            download_audio(url=youtube_link, link_info_dict=link_info_dict, save_path=save_path, overwrite=overwrite)


def divide_into_clips(split='train', raw_music_path='../Dataset/raw_music/', clip_duration=30, stride=15):
    raw_music_full_path = Path(raw_music_path) / split
    all_wav_files = raw_music_full_path.glob('*.wav')
    
    for wav_file in all_wav_files:
        json_file_path = wav_file.with_suffix('.json')
        audio, sr = audio_read(wav_file)
        
        # Make mono
        audio = audio.mean(0)
        points_per_clip = clip_duration * sr
        step_size = stride * sr
        total_clips = int(np.ceil((len(audio) - points_per_clip) / step_size)) + 1
        for i in range(total_clips):
            start_point = i * step_size
            end_point = start_point + points_per_clip
            audio_clip = audio[start_point:end_point]
            
            clip_name = wav_file.parent / f"{wav_file.stem}_{i+1}"
            new_clip_path = clip_name.with_suffix('.wav')
            new_json_path = clip_name.with_suffix('.json')
            shutil.copy(json_file_path, new_json_path)
            sf.write(new_clip_path, audio_clip, sr)
        os.remove(wav_file)
        os.remove(json_file_path)
        print('Done:', wav_file)


def prepare_attributes(split='train', core_music_folder='../Dataset/raw_music/'):
    music_folder = Path(core_music_folder) / split
    music_files = music_folder.glob('*.wav')
    
    for music_path in music_files:
        music_file_name = music_path.stem
        json_file_path = music_path.with_suffix('.json')

        with open(json_file_path, "r") as link_info_json_file:
            link_info_dict = json.load(link_info_json_file)
            
        attribute_dict = {}
        attribute_dict["key"] = ""
        attribute_dict["artist"] = link_info_dict["artist"]
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
        attribute_dict["label"] = link_info_dict["label"]

        with open(json_file_path, "w") as attr_json_file:
            json.dump(attribute_dict, attr_json_file)
            
        print('\033[1mSaved Json file for:\033[0m', music_file_name)


def fill_json_split(split='train', core_music_folder='../Dataset/raw_music/', split_jsonl_path='egs/', essentia_weights_path='../Dataset/essentia_weights/', n_best_preds=3):
    split_jsonl_main_path = Path(split_jsonl_path) / split
    split_jsonl_full_path = split_jsonl_main_path / 'data.jsonl'
    
    raw_music_full_path = Path(core_music_folder) / split 
    all_music_files = raw_music_full_path.glob('*.wav')
    
    if not split_jsonl_main_path.exists():
        os.makedirs(split_jsonl_main_path)
        
    with open(split_jsonl_full_path, 'w') as split_jsonl_file:
        for music_file in all_music_files:
            music_jsonl_info = fill_json(music_file, n_best_preds=n_best_preds, essentia_weights_path=essentia_weights_path)
            music_jsonl_str = json.dumps(music_jsonl_info)
            split_jsonl_file.write(music_jsonl_str + '\n')


def fill_json(music_file, n_best_preds=3, essentia_weights_path = '../Dataset/essentia_weights/'):
    music_file_str = str(music_file)
    json_file_path = music_file.with_suffix('.json')
    
    audio, sr = audio_read(music_file)
    audio = audio[0]
    sec_len = len(audio)/sr

    with open(json_file_path, 'r') as unfilled_json_file:
        json_data = json.load(unfilled_json_file)

    labeling_type = json_data['label']

    if labeling_type == 'essentia':
        music_info = get_essentia_features(audio_filename=music_file_str, n_best_preds = n_best_preds, weights_folder=essentia_weights_path)
    else:
        music_info = custom_labeler(labeling_type, n_best_preds=n_best_preds)
        
    json_data['duration'] = sec_len
    json_data['sample_rate'] = sr
    json_data['genre'] = music_info['genres']
    json_data['instrument'] = ', '.join(music_info['instruments'])
    json_data['moods'] = music_info['moods'].split(', ')

    with open(json_file_path, "w") as filled_json_file:
            json.dump(json_data, filled_json_file)

    jsonl_data = {}

    jsonl_data['path'] = music_file_str
    jsonl_data['duration'] = sec_len
    jsonl_data['sample_rate'] = sr
    jsonl_data['amplitude'] = None
    jsonl_data['weight'] = None
    jsonl_data['info_path'] = None

    return jsonl_data

    print('Done:', music_file)

def fill_descriptions(split='train', core_music_folder='../Dataset/raw_music/', llm_checkpoint = 'ericzzz/falcon-rw-1b-instruct-openorca'):
    split_full_path = Path(core_music_folder) / split 
    json_files = split_full_path.glob('*.json')
    description_generator = DescriptionGenerator(checkpoint = llm_checkpoint)

    for json_file in json_files:
        with open(json_file, 'r') as unfilled_json:
            json_data = json.load(unfilled_json)
        description = description_generator.generate(json_data)
        json_data['description'] = description

        with open(json_file, "w") as filled_json:
            json.dump(json_data, filled_json)
            
        print('\nDone:', json_file)
        print('Description:', description)
        print('-'*50)


def custom_labeler(label, n_best_preds = 3):
    result_dict = {} 
    if label=='duduk':
        result_dict['genres'] = 'Armenian, Armenian Folk'
        result_dict['instruments'] = ['duduk']   
        result_dict['moods'] = ', '.join(random.sample(['melancholic', 'emotional', 'dramatic', 'relaxing', 'sad'], n_best_preds))
    return result_dict