from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn.functional as F
from pathlib import Path
import soundfile
import json
import os

from audiocraft.models.loaders import load_lm_model_ckpt, load_compression_model_ckpt
from audiocraft.utils.utils import get_loader, dict_from_config


from audiocraft.solvers.builders import get_audio_datasets, DatasetType
from audiocraft.models.builders import get_compression_model


from audiocraft.data.audio import audio_read
from audiocraft.data.audio_dataset import SegmentInfo
from audiocraft.data.music_dataset import MusicInfo, augment_music_info_description
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.info_audio_dataset import AudioInfo

from audiocraft.modules.conditioners import (
    T5Conditioner, 
    ConditioningProvider,
    WavCondition, 
    BaseConditioner,
    JointEmbedCondition
)

from audiocraft.models.encodec import EncodecModel


class PreprocessData:
	"""
	class PreprocessData is used for converting audio to audio tokens and descriptions to text embeddings.

	"""

	DATASET_TYPE = DatasetType.MUSIC


	def __init__(self, cfg, checkpoint='facebook/musicgen-small'):
		"""
		parameters:

		- cfg: config file of the checkpoint
		- checkpoint: default checkpoint
		- lm_model_ckpt: loaded checkpoint of the LM model for the default model
		- compression_ckpt: loaded checkpoint of the compression model for the default model
		- target_sr: Target sampling rate to change the audio to (32,000 in this case)
		- target_channels: Target channels to change the audio to (1 in this case, same as mono)
		- datasets: The datasets with their metadata extracted from the egs/{split} JSONL file, where split is train, valid, etc.
		- condition_provider: Text conditioner, same as T5
		- encodec_model: Audio tokenizer, compression model, same as EnCodec
		- processing_info: Instance for the class ProcessingInfo providing info about processing

		"""
		self.cfg = cfg
		self.checkpoint = checkpoint

		self.lm_model_ckpt = load_lm_model_ckpt(self.checkpoint)
		self.compression_ckpt = load_compression_model_ckpt(self.checkpoint)

		self.target_sr = self.cfg.sample_rate
		self.target_channels = self.cfg.channels
		self.segment_duration = self.cfg.dataset.segment_duration
		self.datasets = self.build_datasets()
		self.condition_provider = self.load_conditioner()
		self.encodec_model = self.load_encodec()

		self.processing_info = ProcessingInfo()

	def build_datasets(self, dataset_type = DATASET_TYPE):
		"""
		Function for building the datasets from JSONL file containing the metadata See solvers.builders.get_audio_datasets for more info.
	
		"""
		return get_audio_datasets(self.cfg, dataset_type=dataset_type)


	def read_music(self, meta, index):
		"""
		Function for reading the audio and changing the sampling rate to the target sampling rate, as well as the number of channels to the target channels.

		parameters:

		- meta: The list full of audio meta pieces taken from egs/{split} jsonl file
		- index: The index of the meta file

		returns:
		- Returns the converted audio to target sr and channels. 

		"""
		out, sr = audio_read(meta[index].path)
		out = convert_audio(out, sr, self.target_sr, self.target_channels)
		return out

	def read_info(self, loader, meta_info, meta, index):
		"""
		Function for reading the info about the music piece from the meta data

		parameters:
		- loader: The data loader of the split: train, valid, etc.
		- meta_info: Info taken from the meta 
		- meta: The list full of audio meta pieces taken from egs/{split} jsonl file

		"""

		# Music info json file
		music_info_path = Path(meta[index].path).with_suffix('.json')

		# Info from the meta file
		info_data = meta_info.to_dict()

		if Path(music_info_path).exists():
			with open(music_info_path, 'r') as json_file:
				# Load the json file
				music_data = json.load(json_file)

				# Update the json file with the metadata
				music_data.update(info_data)

				# Make it a MusicInfo class
				music_info = MusicInfo.from_dict(music_data, fields_required=loader.dataset.info_fields_required)
			
			# A default code piece from the MusicGen, not sure if this changes something
			if loader.dataset.paraphraser is not None:
				music_info.description = loader.dataset.paraphraser.sample(music_info.meta.path, music_info.description)

			# A default code piece from the MusicGen, not sure if this changes something
			if loader.dataset.merge_text_p:
				music_info = augment_music_info_description(music_info, loader.dataset.merge_text_p, loader.dataset.drop_desc_p, loader.dataset.drop_other_p)
        
		else:
			music_info = MusicInfo.from_dict(info_data, fields_required=False)

		return music_info

	def load_conditioner(self):
		"""
		Function for loading the text conditioner.

		"""

		# Get conditioner part of the config
		conditioner_cfg = getattr(self.cfg, 'conditioners')

		# Get dictionary from the config
		dict_cfg = {} if conditioner_cfg is None else dict_from_config(conditioner_cfg)

		# Pop some parameters
		conditioners = {}
		condition_provider_args = dict_cfg.pop('args', {})
		condition_provider_args.pop('merge_text_conditions_p', None)
		condition_provider_args.pop('drop_desc_p', None)
	
		# Get description part of the conditioner config
		cond_cfg = dict_cfg['description']
		model_type = cond_cfg['model']
		model_args = cond_cfg[model_type]

		# Instantiate the T5 Conditioner
		conditioners['description'] = T5Conditioner(output_dim=self.cfg.transformer_lm['dim'], device=self.cfg.device, **model_args)

		# Return the best state
		return self.load_conditioner_state_dict(conditioners, condition_provider_args)


	def load_conditioner_state_dict(self, conditioners, condition_provider_args):
		"""
		Function for loading the T5 text conditioner best state 

		"""
		state = {
		            'best_state': {
		                'model': self.lm_model_ckpt['best_state'],
		            },
		        }

		# Best linear layer from 768 -> 1024
		output_proj_weight = state['best_state']['model'].pop('condition_provider.conditioners.description.output_proj.weight')
		output_proj_bias = state['best_state']['model'].pop('condition_provider.conditioners.description.output_proj.bias')

		conditioners['description'].output_proj.load_state_dict({'weight': output_proj_weight, 'bias': output_proj_bias})

		# Instantiate Conditioning Provider from T5 Conditioner and Linear Layer
		condition_provider = ConditioningProvider(conditioners, device=self.cfg.device, **condition_provider_args).to(self.cfg.device)

		# Switch to eval mode
		return condition_provider.eval()

	def load_encodec(self):
		"""
		Function for loading EnCodec model

		"""
		# Get the compression model config
		compression_cfg = OmegaConf.create(self.compression_ckpt['xp.cfg'])

		# Load the compression model from the config
		encodec_model = get_compression_model(compression_cfg) 

		# Load the best state
		encodec_model.load_state_dict(self.compression_ckpt['best_state'])
		encodec_model

		#Switch to eval mode
		return encodec_model.eval()



	def run(self, time_shift=15.0, data_split='train', save_path='../../dataset/'):
		"""

		Function for processing a full split (train, valid, etc.) and saving it in a directory

		"""
		# Get the split from the datasets
		loader = self.datasets[data_split]

		# Get the meta list
		file_meta = loader.dataset.meta
		self.processing_info.preprocess_start(file_meta, data_split)
		seek_time = 0.0

		# For meta file
		for it, mus in enumerate(file_meta):

			# Read the music
			out = self.read_music(file_meta, it)

			# Get number of frames
			n_frames = out.shape[-1]

			# Get target frames
			target_frames = int(self.segment_duration * self.target_sr)

			if loader.dataset.pad:

				# Pad the audio to target frames if necessary
				padding_amount = max(0, target_frames - n_frames)
				self.processing_info.padding_info(int(padding_amount/self.target_sr), target_frames)
				out = F.pad(out, (0, padding_amount))
				n_frames = out.shape[-1]



			if loader.dataset.return_info:
				# Get the meta info with AudioInfo class.
				segment_info = SegmentInfo(file_meta[it], seek_time, n_frames=n_frames,
					total_frames=target_frames, sample_rate=self.target_sr, channels=out.shape[0])
				meta_info = AudioInfo(**segment_info.to_dict())

			# Get the music info
			music_info = self.read_info(loader, meta_info, file_meta, it)

			# Default code from MusicGen for other uses
			music_info.self_wav = WavCondition(
				wav = out[None], length=torch.tensor([meta_info.n_frames]),
				sample_rate = [meta_info.sample_rate], path=[meta_info.meta.path], seek_time=[meta_info.seek_time])

			# Default code from MusicGen for other uses
			for att in loader.dataset.joint_embed_attributes:
				att_value = getattr(music_info, att)
				joint_embed_cond = JointEmbedCondition(
					out[None], [att_value], torch.tensor([meta_info.n_frames]),
        			sample_rate = [meta_info.sample_rate], path=[meta_info.meta.path], seek_time=[meta_info.seek_time])
				music_info.joint_embed[att] = joint_embed_cond


			# If audio segment is longer than 30, than slicing is needed (cut to 30-second intervals)
			slicing = False if n_frames/self.target_sr <= self.segment_duration else True
			self.processing_info.slicing_info(slicing)

			if save_path:
				pure_path = file_meta[it].path
				file_name = os.path.splitext(os.path.basename(pure_path))[0]

			if slicing:
				shift_samples = int(time_shift * self.target_sr)

				# For each slice cut 30-second intervals
				for i, start in enumerate(range(0, n_frames - target_frames + 1, shift_samples)):
					end = start + target_frames
					clip = out[:, start:end]

					# Tokenize audio and tokenize text
					condition_tensors, audio_tokens, padding_mask = self.prepare_attributes(clip, music_info)

					if save_path:
						file_name_i = file_name + f'_{i+1}'

						# Save the processed files
						self.save(condition_tensors, audio_tokens, padding_mask, data_split=data_split, file_name=file_name_i, save_path=save_path)
				if save_path:
					self.processing_info.save_info(file_name, i=i)


			else:
				# If no slicing, then prepare the 30-second audio clip
				condition_tensors, audio_tokens, padding_mask = self.prepare_attributes(out, music_info)
				if save_path:
					self.save(condition_tensors, audio_tokens, padding_mask, data_split=data_split, file_name=file_name, save_path=save_path)
					self.processing_info.save_info(file_name)
			self.processing_info.mus_end()
		self.processing_info.end_of_info()

	def save(self, condition_tensors, audio_tokens, padding_mask, data_split, file_name, save_path):
		"""
		Function for saving the processed files
		"""

		# Convert the processed files to cpu
		condition_tensors['description'] = tuple(tens.cpu().detach().squeeze(0) for tens in condition_tensors['description'])

		# Change the dimensions (removing the first dimension's 1)
		audio_tokens = audio_tokens.cpu().detach().squeeze(0)
		padding_mask = padding_mask.cpu().detach().squeeze(0)

		# Get the attributes dict that should be saved
		attributes_dict = {'condition_tensors': condition_tensors, 'padding_mask': padding_mask}

		# Save
		path_to_save = save_path + data_split + '/'+ file_name
		if not os.path.exists(path_to_save):
			os.makedirs(path_to_save)
		torch.save(attributes_dict, path_to_save + '/attributes.pt')
		torch.save(audio_tokens, path_to_save + '/encodec_encoding.pt')


	def prepare_attributes(self, wav, info):
		"""
		Function for preparing the attributes

		"""

		# Convert to GPU
		wav = wav.to(self.cfg.device)

		if wav.dim()==2:
			wav = wav[None]

		# Prepare and tokenize the descriptions
		attributes = info.to_condition_attributes()
		tokenized = self.condition_provider.tokenize([attributes])

		# Encode the audio into audio tokens
		with torch.no_grad():
			audio_tokens, scale = self.encodec_model.encode(wav)
			assert scale is None, "Scaled compression model not supported with LM."

		# Forward pass through linear layer
		condition_tensors = self.condition_provider(tokenized)

		# Get the padding mask
		padding_mask = torch.ones_like(audio_tokens, dtype = torch.bool, device=audio_tokens.device)

		return condition_tensors, audio_tokens, padding_mask




class ProcessingInfo:
	def preprocess_start(self, file_meta, split):
		print("="*100)
		print(f'Starting preprocessing of the data split -> {split}')
		print(f'With in total of {len(file_meta)} music files')
		print("="*100)

	def padding_info(self, padding_amount, target_frames):
		print(f'\nPadding = {padding_amount} sec.')

	def slicing_info(self, slicing):
		print(f'Slicing: {slicing}')

	def mus_end(self):
		print('\n')
		print('-'*50)

	def save_info(self, filename, i=0):
		if i:
			print(f'\nIn total of {i} music clips where extracted and successfully saved from the file {filename}.')
		else:
			print(f'\nThe file {filename} was successfully saved.')

	def end_of_info(self):
		print('\nEnd of the processing\n')
		print("="*100)






















