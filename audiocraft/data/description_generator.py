import json
from transformers import AutoTokenizer
import transformers
import torch

class DescriptionGenerator:
    def __init__(self, checkpoint='ericzzz/falcon-rw-1b-instruct-openorca'):
        self.checkpoint = checkpoint
        self.pipeline = self.load_pipeline()
        
    def load_pipeline(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        pipeline = transformers.pipeline(
           'text-generation',
           model=self.checkpoint,
           tokenizer=tokenizer,
           torch_dtype=torch.bfloat16,
           pad_token_id=50256,
           device = device

        )
        return pipeline

    def gen_default_description(self, json_data):
        moods = json_data['moods']
        genre = json_data['genre']
        instrument = json_data['instrument']

        default_description = f'Generate a music that has the following genres: {genre}. The following Instruments: {instrument}. The following Moods: ' + ', '.join(moods) 
        return default_description


    def get_manual_description(self, json_data):
        return json_data['manual_description']
        
    
    def create_prompt(self, json_data):
        moods = json_data['moods']
        genre = json_data['genre']
        instrument = json_data['instrument']

        prompt = f'Generate a moderate length and very creative description using all the following independent keywords: {genre}, {instrument}, ' + ', '.join(moods)
        
        return prompt


    def gen_creative_description(self, data, rep_penalty=1.05):

        creative_prompt = self.create_prompt(data)

        system_message = "Be very creative. Use only words that are connected to the description, do not add unnecessary keywords."
        prompt = f'<SYS> {system_message} <INST> {creative_prompt} <RESP> A music that has'
        
        response = self.pipeline(
           prompt, 
           max_length=200,
           truncation=True,
           repetition_penalty=rep_penalty
        )

        prompt_generation = response[0]['generated_text']

        creative_description = self.extract_generation(prompt_generation)
        
        return creative_description

    def extract_generation(self, text):
        resp_split = text.split('<RESP> ')
        return resp_split[-1]
        