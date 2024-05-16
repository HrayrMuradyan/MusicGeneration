# Armenian Music Generation

Armenian Music Generation model is a fine-tuned model based on [AudioCraft](https://github.com/facebookresearch/audiocraft/tree/main), which is a PyTorch library for audio generation, specifically, on MusicGen ([Simple and controllable model for music generation](https://arxiv.org/abs/2306.05284)).  

We modified the codes for more efficient training pipeline implementation and fine-tuned the MusicGen Small model on ~120 hours of Armenian music extracted from YouTube of diverse genres, moods and instruments.

The project is completed as a final, Capstone project for the American University of Armenia, with Hrayr Muradyan as the primary contributor and Karlos Muradyan serving as the supervisor.

## Examples

<b>Prompt:</b> A sad and melancholic play on duduk. An Armenian instrumental music that evokes relaxation, calmness accompanied by sorrow and uncheerfulness. It makes the listener think about life, fall into deep contemplation and reevaluate the past, showing the old heritage of Armenia.

[Link to audio file #1](https://www.youtube.com/watch?v=aAxdNWj_KfY)

<b>Prompt:</b> A music that has the following genres: Armenian folk. The following Instruments: klarnet, percussion, synthesizer, drums, bass. The following Moods: happy, energetic, melodic.

[Link to audio file #2](https://www.youtube.com/watch?v=UGDMU5I1SLE)

<b>Prompt:</b> The following traditional Armenian dance music is played using zurna, drums, percussion and synthesizer. The beautiful blend of this magic instruments guarantees an active, dancing mood that fills the air with an irresistible energy, prompting all to sway and groove to the uplifting beat.

[Link to audio file #3](https://www.youtube.com/watch?v=Wz5eKHjeQqU)

## Documentations

All necessary documentations of the repository can be found in the [additional_tools/documentations/](./additional_tools/documentations/) folder.

## Installation

The whole installation documentation can be found in the [Training and Environment Documentation File](./additional_tools/documentations/Training%20and%20Environment%20Documentation.pdf). A step-by-step explanation of how to setup the environment and proceed with training.

## Data Preparation

Data preparation documentation used in our approach is fully described in the [Data Documentation](./additional_tools/documentations/Data%20Documentation.pdf). 

## Fine-tuned checkpoint

The weights of the model can be found in the [additional_tools/checkpoints/](./additional_tools/checkpoints/)
folder. The weights are uploaded to a drive, from which they can be easily downloaded.

## Dataset

The dataset is also publicly available in the folder [additional_tools/youtube_music_links/](./additional_tools/youtube_music_links/). Refer to the <b>Data Preparation</b> heading to find out how to extract the data.

## Flask Application   

Additionally, we have prepared an easy tool which allows anyone to use the model in an user-friendly interface. The documentation for running the Flask application can be found in [Flask App Documentation](./additional_tools/documentations/Flask%20App%20Documentation.pdf).

Here is the screenshow of how it looks like:

<img width="866" alt="Capture" src="https://github.com/HrayrMuradyan/MusicGeneration/assets/82998878/f140e261-d3b3-4396-997e-2788c909970f">



## Citation

We used the source code for the AudioCraft framework, specifically MusicGen, Simple and Controllable Music Generation, developed by Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez. The paper was presented at the Thirty-seventh Conference on Neural Information Processing Systems in 2023.

