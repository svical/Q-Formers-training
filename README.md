# Q-Formers-training
Q-Former is a lightweight transformer module that learns a compact, task-adaptable representation of an image by querying pre-extracted vision features with a small set of learnable tokens. It makes language model have the ability of process image input, bridging Vision and Language together. This repository is pytorch code for training a mini Q-Former model for image captioning task. 

<br>
<div align="center">
<img src="assets/Picture1.png" alt="bridge" width="70%">
</div>
<br>

## Quick start
An [Anaconda](https://docs.anaconda.com/anaconda/install/) environment is recommanded for this project. The first step is to create a virtual environment, as shown below (named `Qformers`).
```
conda create -n Qformers python=3.10
conda activate Qformers
```
Then, install the required packages.
```
pip install -r requirements.txt
```
Download models from huggingface. We use [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) model as the image encoder and [T5](https://huggingface.co/docs/transformers/en/model_doc/t5) and [Llama 3.1 3B instruct](https://huggingface.co/docs/transformers/en/model_doc/llama) as the downstream LLMs. You might need to apply for a grant in huggingface to use Llama models.
```
cd models
git lfs clone https://huggingface.co/openai/clip-vit-base-patch32
git lfs clone https://huggingface.co/google-t5/t5-base
git lfs clone https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

At last download datasets for training
```
cd datasets
git lfs clone https://huggingface.co/datasets/shirsh10mall/Image_Captioning_GCC_Embeddings
```

## Training and inferencing

I have several trials on different training configurations. You can just pick one of the script for training.
```
python3 training_simpleT5.py
```
The loss curve chart and model weights will appear in `./checkpoints_t5`. If you don't want to train, you can download a pretrained weight file from [here](https://drive.google.com/file/d/1HhwibTsMnBYVFfGYJmdsaNfx0jA42UVY/view?usp=drive_link).

For inferencing, please run this code.
```
python3 t5_ference.py
```
