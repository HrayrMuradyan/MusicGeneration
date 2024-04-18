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


    def create_prompt(self, json_data):
        moods = json_data['moods']
        genre = json_data['genre']
        instrument = json_data['instrument']
        
        prompt = f'Generate a description containing all the following genres: ({genre}). All the following Instruments: ({instrument}). All the following Moods: (' + ', '.join(moods) + ')'
        
        return prompt


    def generate(self, data, rep_penalty=1.05):

        prompt_text = self.create_prompt(data)

        system_message = 'Do not miss any of the keywords in the brackets. Provide no additional information.'
        prompt = f'<SYS> {system_message} <INST> {prompt_text} <RESP> A music that has'
        
        response = self.pipeline(
           prompt, 
           max_length=200,
           repetition_penalty=rep_penalty
        )

        prompt_generation = response[0]['generated_text']
        
        return self.extract_generation(prompt_generation)

    def extract_generation(self, text):
        resp_split = text.split('<RESP> ')
        return resp_split[-1]
        