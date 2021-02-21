# Rainforest Connection Species Audio Detection - 54th place
## https://www.kaggle.com/c/rfcx-species-audio-detection

In order to quickly train in Google Colab, audio files were locally converted into melspectograms using the generate_mel_spec script and then stored as a HDF5 file which was then read into memory for traning and inference (reading from google drive while training was about 5x slower than reading from RAM).

### Audio Encoding
- Melspectogram
- 32kHz sampling rate
- 128 mels
- Mel power 2

### Base Model
#### Training
- Pretrained efficientNet-B4, 2x64 fully connected layers with 0.3 dropout
- Input size 380x380
- No image augmentations
- 200 epochs
- Batch size of 32
- Random 4 second crop from 6 second window around sample (hard labels)
- 5-fold cross validation

#### Inference
1. Projections made on every 4 second window for the 60 second sample using a sliding window of 1 second.
2. These results were then adjusted by using a 3 second running average to smoothen out the estimates (raised LB score by roughly 0.005)
3. Max value accross all samples and folds for each class gave the final score for each recording

### Final Blend
- EfficientNet-B4
- EfficientNet-B3
- ResNest50
- ResNest101

