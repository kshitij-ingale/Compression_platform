
# CompressGAN
## Training framework for image compression using Generative Adversarial Networks Approach
Deep learning based image compression techniques focus on learning distribution behind images to extract latent features from images enabling extreme compression ratios. One such approach involves encoding images to latent features, quantizing the vector for lossy compression and storing this compressed vector. In order to reconstruct the image, decoder network can be used which is trained with adversarial loss framework. This implementation is based on the paper [Generative Adversarial Networks for Extreme Learned Image Compression](https://arxiv.org/abs/1804.02958)  by Agustsson et. al.


## Repository Structure:
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

## Inference
Model inference can be done as compression as well as reconstruction of input image which generates quantized vector file, reconstructed image and comparison of reconstructed image against input image. For single image inference, provide path to image while for multiple images provide the path to input image directory using the following command
```
python src/inference.py -r *model checkpoint location* -path *input image* -o *output directory*
```
Another inference can be decompressing binary file consisting of quantized latent vector which can be done by the following command
```
python src/inference.py -r *model checkpoint location* -c *binary file* -o *output directory*
```

## Training model
The model configuration file config.py determines the parameters for the compression, input image attributes and model characteristics. If preprocessing is required, preprocess.py can be used (currently it resizes input image to dimensions indicated by input attributes in config.py). The model input is in the form of hdf5 file consisting of image paths under label 'path'. This can be created by train.py by providing image directory to path argument of train.py. Finally, the model can be trained using the following command
```
python src/train.py -r *model checkpoint location* -path *input images directory*
```

## Experimentation:
- For better results and faster training, perceptual loss inspired from [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) by Johnson et al. was incorporated into the training process. The pretrained model for this loss was sourced from pre-built VGG19 model from tf.keras which provided the feature map for input image and reconstructed image. The RMSE for these feature maps constituted the additional term in the loss function.

- Dataset size was compared for performance on faces dataset. The model was trained using 12000 images during first iteration and using around 39000 images during the second iteration. It was observed that performance (PSNR) improved by around 20% for 3x training data size.

- The impact of quantization factors like number of latent factors and number of quantization centers on model performance was analyzed indicating almost constant trend in PSNR and increasing number of bits to encode the image to latent vectors. The configuration of 8 latent factors and 5 quantization centers seemed to be slightly optimal as compared to the other 2 configurations

- Another experiment around encoding techniques for latent factors was considered. Initially, block encoding technique was used to encode the quantized latent vector, however, it was observed adaptive arithmetic encoding technique provided better compression with file size decreasing by around 70%

## Results:

