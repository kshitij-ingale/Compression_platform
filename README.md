
# CompressGAN
## Training framework for image compression using Generative Adversarial Networks Approach

## Structure:
- **checkpoints** : Directory for saving model checkpoints during training or inference
- **data** : Directory for storing images to be compressed (example dataset provided in this directory)
- **images** : Directory for storing images for observing results (example images provided in this directory)
- **output** : Directory for storing output images, compressed files and comparisons with original images
- **src** : Directory for all source code files
- **tensorboard** : Directory for saving tensorboard summary

## Setup
Clone repository using the following command
```
git clone https://github.com/kshitij-ingale/Compression_platform
```

## Requirements
The codebase has some dependencies as specified in the requirements.txt, which can be used to install using pip or a new virtual environment can be created by providing the requirements file

#### Installation
To install the package above, pleae run:
```shell
pip install -r requirements.txt
```

## Compress single image
- Model Inference can be done on a single image or a batch of images
- For single image inference, download the pretrained model and run the following
```
python src/singlefile.py -r *checkpoint location* -i *input image* -o *output*
```
- For multiple images,
```
python src/inference.py -r *model checkpoint location* -path *input images directory* -o *output directory*
```
## Training model
- The model can be trained by running the following
```
python src/train.py -r *model checkpoint location* -path *input images directory*
```
