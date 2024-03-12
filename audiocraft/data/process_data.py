from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn.functional as F
import typing as tp
from pathlib import Path
import soundfile
import json
import os

from audiocraft.models.loaders import load_lm_model_ckpt, load_compression_model_ckpt
from audiocraft.utils.utils import get_loader, dict_from_config


from audiocraft.solvers.builders import get_audio_datasets, DatasetType
from audiocraft.models.builders import get_encodec_autoencoder, get_quantizer


from audiocraft.data.audio import audio_read
from audiocraft.data.audio_dataset import SegmentInfo
from audiocraft.data.music_dataset import MusicInfo
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.info_audio_dataset import AudioInfo

from audiocraft.modules.conditioners import (
    T5Conditioner, 
    ConditioningProvider,
    WavCondition, 
    BaseConditioner
)

from audiocraft.models.encodec import EncodecModel


class PreprocessData:

	DATASET_TYPE = DatasetType.MUSIC


	def __init__(self, cfg, checkpoint='facebook/musicgen-small'):
		self.cfg = cfg
		self.checkpoint = checkpoint

		self.lm_model_ckpt = load_lm_model_ckpt(self.checkpoint)
		self.compression_ckpt = load_compression_model_ckpt(self.checkpoint)

		self.target_sr = self.cfg.sample_rate
		self.target_channels = self.cfg.channels
		self.segment_duration = self.cfg.dataset.segment_duration
		self.datasets = self.build_datasets()
		self.load_conditioner()
		self.load_encodec()

		self.processing_info = ProcessingInfo()

	def build_datasets(self, dataset_type = DATASET_TYPE):
		return get_audio_datasets(self.cfg, dataset_type=dataset_type)


	def read_music(self, meta, index):
		out, sr = audio_read(meta[index].path)
		out = convert_audio(out, sr, self.target_sr, self.target_channels)
		return out

	def read_info(self, loader, meta_info, meta, index):
		music_info_path = Path(meta[index].path).with_suffix('.json')
		info_data = meta_info.to_dict()

		if Path(music_info_path).exists():
			with open(music_info_path, 'r') as json_file:
				music_data = json.load(json_file)
				music_data.update(info_data)
				music_info = MusicInfo.from_dict(music_data, fields_required=loader.dataset.info_fields_required)
			
			if loader.dataset.paraphraser is not None:
				music_info.description = loader.dataset.paraphraser.sample(music_info.meta.path, music_info.description)

			if loader.dataset.merge_text_p:
				music_info = augment_music_info_description(music_info, loader.dataset.merge_text_p, loader.dataset.drop_desc_p, loader.dataset.drop_other_p)
        
		else:
			music_info = MusicInfo.from_dict(info_data, fields_required=False)

		return music_info

	def load_conditioner(self):
		conditioner_cfg = getattr(self.cfg, 'conditioners')
		dict_cfg = {} if conditioner_cfg is None else dict_from_config(conditioner_cfg)

		conditioners = {}
		condition_provider_args = dict_cfg.pop('args', {})
		condition_provider_args.pop('merge_text_conditions_p', None)
		condition_provider_args.pop('drop_desc_p', None)

		for cond, cond_cfg in dict_cfg.items():
			model_type = cond_cfg['model']
			model_args = cond_cfg[model_type]
			conditioners[str(cond)] = T5Conditioner(output_dim=self.cfg.transformer_lm['dim'], device=self.cfg.device, **model_args)

		self.load_conditioner_state_dict(conditioners, condition_provider_args)

		# cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg.classifier_free_guidance.training_dropout)
		# att_dropout = AttributeDropout(p=cfg.attribute_dropout)


	def load_conditioner_state_dict(self, conditioners, condition_provider_args):
		state = {
		            'best_state': {
		                'model': self.lm_model_ckpt['best_state'],
		            },
		        }

		output_proj_weight = state['best_state']['model'].pop('condition_provider.conditioners.description.output_proj.weight')
		output_proj_bias = state['best_state']['model'].pop('condition_provider.conditioners.description.output_proj.bias')

		conditioners['description'].output_proj.load_state_dict({'weight': output_proj_weight, 'bias': output_proj_bias})

		self.condition_provider = ConditioningProvider(conditioners, device=self.cfg.device, **condition_provider_args).to(self.cfg.device)

	def load_encodec(self):
		
		compression_cfg = OmegaConf.create(self.compression_ckpt['xp.cfg'])

		kwargs = dict_from_config(getattr(compression_cfg, 'encodec'))

		encoder_name = kwargs.pop('autoencoder')
		quantizer_name = kwargs.pop('quantizer')

		encoder, decoder = get_encodec_autoencoder(encoder_name, compression_cfg)
		quantizer = get_quantizer(quantizer_name, compression_cfg, encoder.dimension)

		frame_rate = kwargs['sample_rate'] // encoder.hop_length
		renormalize = kwargs.pop('renormalize', False)

		kwargs.pop('renorm', None)

		self.encodec_model = EncodecModel(encoder, decoder, quantizer, 
						frame_rate=frame_rate, renormalize=renormalize, **kwargs).to(compression_cfg.device)

		self.encodec_model.load_state_dict(self.compression_ckpt['best_state'])
		self.encodec_model.eval()



	def run(self, time_shift=15.0, data_split='train', save_path='../../dataset/tensors/'):
		loader = self.datasets[data_split]
		file_meta = loader.dataset.meta
		self.processing_info.preprocess_start(file_meta, data_split)
		seek_time = 0.0
		for it, mus in enumerate(file_meta):
			out = self.read_music(file_meta, it)

			n_frames = out.shape[-1]
			# target_frames = int(self.segment_duration * self.target_sr)
			target_frames = int(self.segment_duration * self.target_sr)

			if loader.dataset.pad:
				padding_amount = -min(0, n_frames - target_frames)
				self.processing_info.padding_info(int(padding_amount/self.target_sr), target_frames)
				out = F.pad(out, (0, padding_amount))
				n_frames = out.shape[-1]



			if loader.dataset.return_info:
				segment_info = SegmentInfo(file_meta[it], seek_time, n_frames=n_frames,
					total_frames=target_frames, sample_rate=self.target_sr, channels=out.shape[0])
				meta_info = AudioInfo(**segment_info.to_dict())

			music_info = self.read_info(loader, meta_info, file_meta, it)

			music_info.self_wav = WavCondition(
				wav = out[None], length=torch.tensor([meta_info.n_frames]),
				sample_rate = [meta_info.sample_rate], path=[meta_info.meta.path], seek_time=[meta_info.seek_time])

			for att in loader.dataset.joint_embed_attributes:
				att_value = getattr(music_info, att)
				joint_embed_cond = JointEmbedCondition(
					out[None], [att_value], torch.tensor([meta_info.n_frames]),
        			sample_rate = [meta_info.sample_rate], path=[meta_info.meta.path], seek_time=[meta_info.seek_time])
				music_info.joint_embed[att] = joint_embed_cond


			slicing = False if n_frames/self.target_sr <= self.segment_duration else True
			self.processing_info.slicing_info(slicing)

			if slicing:
				shift_samples = int(time_shift * self.target_sr)
				file_name = file_meta[it].path.split('\\')[-1][:-4]
				for i, start in enumerate(range(0, n_frames - target_frames + 1, shift_samples)):
					end = start + target_frames
					clip = out[:, start:end]

					condition_tensors, audio_tokens, padding_mask = self.prepare_attributes(clip, music_info)

					file_name_i = file_name + f'_{i+1}'

					if save_path:
						self.save(condition_tensors, audio_tokens, padding_mask, data_split=data_split, file_name=file_name_i, save_path=save_path)
				if save_path:
					self.processing_info.save_info(file_name, i=i)


			else:
				condition_tensors, audio_tokens, padding_mask = self.prepare_attributes(out, music_info)
				if save_path:
					file_name = file_meta[it].path.split('\\')[-1][:-4]
					self.save(condition_tensors, audio_tokens, padding_mask, data_split=data_split, file_name=file_name, save_path=save_path)
					self.processing_info.save_info(file_name)
			self.processing_info.mus_end()
		self.processing_info.end_of_info()

	def save(self, condition_tensors, audio_tokens, padding_mask, data_split, file_name, save_path):
		path_to_save = save_path + data_split + '/'+ file_name
		if not os.path.exists(path_to_save):
			os.makedirs(path_to_save)
		torch.save(condition_tensors, path_to_save + '/condition_tensor.pt')
		torch.save(audio_tokens, path_to_save + '/audio_tokens.pt')
		torch.save(padding_mask, path_to_save + '/padding_mask.pt')



	def prepare_attributes(self, wav, info):
		wav = wav.to(self.cfg.device)

		if wav.dim()==2:
			wav = wav[None]

		attributes = info.to_condition_attributes()
		tokenized = self.condition_provider.tokenize([attributes])

		with torch.no_grad():
			audio_tokens, scale = self.encodec_model.encode(wav)
			assert scale is None, "Scaled compression model not supported with LM."

		condition_tensors = self.condition_provider(tokenized)
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












