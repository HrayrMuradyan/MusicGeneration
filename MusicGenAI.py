import sys
sys.path.insert(0, '../')

import torch
from omegaconf import OmegaConf

from audiocraft.models.loaders import load_lm_model_ckpt, _delete_param, load_compression_model
from audiocraft.models.musicgen import MusicGen
from audiocraft.models.builders import get_lm_model

checkpoint_def = 'facebook/musicgen-small'
checkpoint_trained = '/home/karlos/Documents/workspace/projects/music/trained_models/checkpoint39.th'

if torch.cuda.device_count():
    device = 'cuda'
else:
    device = 'cpu'

cache_dir = None
memory_saver = False

class MusicGenAI:
    def __init__(self):
        self.model = None

    def load_model(self):
        print('Loading model...')
        lm_model_ckpt = load_lm_model_ckpt(checkpoint_trained, cache_dir=cache_dir)
        cfg = OmegaConf.create(lm_model_ckpt['xp.cfg'])
        print('1/5 -> Trained Checkpoint Loaded...')
        lm_model_ckpt_def = load_lm_model_ckpt(checkpoint_def, cache_dir=cache_dir)
        cfg_def = OmegaConf.create(lm_model_ckpt_def['xp.cfg'])
        print('2/5 -> Initial Checkpoint Loaded...')

        if cfg.device == 'cpu':
            cfg.dtype = 'float32'
        else:
            cfg.dtype = 'float16'
        OmegaConf.update(cfg_def, "memory_saver.enable", memory_saver)
        _delete_param(cfg_def, 'conditioners.self_wav.chroma_stem.cache_path')
        _delete_param(cfg_def, 'conditioners.args.merge_text_conditions_p')
        _delete_param(cfg_def, 'conditioners.args.drop_desc_p')

        lm_model = get_lm_model(cfg_def)
        condition_weight = 'condition_provider.conditioners.description.output_proj.weight'
        condition_bias = 'condition_provider.conditioners.description.output_proj.bias'

        lm_model_ckpt['best_state']['model'][condition_weight] = lm_model_ckpt_def['best_state'][condition_weight]
        lm_model_ckpt['best_state']['model'][condition_bias] = lm_model_ckpt_def['best_state'][condition_bias]

        lm_model.load_state_dict(lm_model_ckpt['best_state']['model'])
        lm_model.eval()
        lm_model.cfg = cfg
        print('3/5 -> LM model loaded...')

        compression_model = load_compression_model(checkpoint_def, device=device)
        if 'self_wav' in lm_model.condition_provider.conditioners:
            lm_model.condition_provider.conditioners['self_wav'].match_len_on_eval = True
            lm_model.condition_provider.conditioners['self_wav']._use_masking = False
        print('4/5 -> Compression model loaded...')

        self.model = MusicGen(checkpoint_def, compression_model, lm_model)
        self.model.set_generation_params(duration=20)
        print('5/5 -> MusicGen model Initialized...')


    def generate_music(self, text):
        music = self.model.generate([text])
        return music[0][0].detach().cpu()


