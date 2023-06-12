# Denoising Autoencoder

This project is a first step toward a proof of concept for the implementation of a denoising autoencoder as a preprocessing step for an acoustic guitar tuner. The inspiration for this project came from countless frustrating experiences trying to tune my guitar during orchestra warmups or in other environments where I found myself among other musicians playing thier instruments. The harmonic noise from the other instruments often prevented the tuner from reliably identifying the fundamental frequency of the note played on the guitar, making tuning frustrating, if not impossible altogether. My objective in this project is the explore the possibility of training a neural network to separate an acoustic guitar signal from a mixed signal of an acoustic guitar, harmonic noise from other instruments, and other environmental noise.



# Data

*A more detailed description of the sourced of the data for this project can be found at the bottom of this page.*

### Audio Samples
* **sample rate**: 22050hz
* **sample lengths**: 2^16 samples (~3 seconds)
* **guitar samples**: 2642
    * train size: 2377
    * validation size: 265 (10%)
* **harmonic noise samples**: 2612
* **environmental noise samples**: 2400


### Compiling the Dataset
A random selection of the harmonic and environmental noise samples were added to the guitar audio samples at a random signal to noise ration between 0 and 5. The paths of the original target and noise files for each mixed audio file can be found in the [metadata csv file](./dataset_metadata.csv). A more detailed look at the addition of noise to recordings of an acoustic guitar can be found in the [splitter.py](./splitter.py) and [noisifier.py](./noisifier.py) preprocessing modules. My own implementation of the tools in these modules can be found in the [preprocessing notebook](./preprocessing.ipynb).

### Spectrograms
Rather than working with raw waveforms, I decided to work with spectrograms obtained using the short time fourier transform, which takes quantized audio signal as its input (e.g. a long 1-dimensional array, around 65000 samples long in the case of my audio samples) and outputs a 2-dimensional array of complex numbers. Rather than work with complex numbers directly, I converted the complex spectrograms into their corresponding arrays of polar coordinates. This Allowed me to work with time-frequency information represented as a real-valued magnitude spectrogram and with phase information represented as the thetas for the corresponding angle of the polar coordinates. This also allowed the phase to be represented by a single real values 2d array, rather than a separate 2d array for each of the real and imaginary components of the phase information. The magnitude spectrogram was scaled to decibles to make their interpretation more intuitive and to increase the resolution at the frequencies that most directly apply to guitar tuning. These decible values were then min-max scaled between -100 and 100 to allow for the full 96db range before 16bit audio begins clipping in unseen audio. The angle spectrograms were scaled between -1 and 1. The magnitude and phase spectrograms are regularly referred to as rho and thete, reflecting their correspondence to the magnitude and angle that they represent as polar coordinates.

# Generative vs Subtractive Denoising

### Spectrogram Masks
Most audio source separation tasks, such as stemming, denoising, and speech enhancement, rely on some kind of subtractive method to remove unwanted sound from a given audio signal. These are most often models that either predict a binary mask or a ratio mask that is applied to a time-frequency spectrogram. In the case of a binary mask, an array of the size of the input spectrogram is populated with ones and zeros according to the model's prediction. This reduces any values that the model predicts to be noise to zero and leaves all other values unaltered. A ratio mask is similar, though each value in its array is set to some float between zero and one, rescaling the audio such that only the sound that the model preicts as the desired target is audible. The approach I have chosen to explore takes a difffernt approach. [Here](https://source-separation.github.io/tutorial/basics/tf_and_masking.html) is a link to an excellent introduction to music source separation using binary and ratio masks.

### Generative Denoising
Rather than estimating a mask to applied to the noisy audio signal, I will be taking the noisy audio as an input and estimating the original target audio directly using an Autencoder architecture. The autoencoder is composed of two parts, an encoder and a decoder. On a high level an encoder, works by performing non-linear dimensionality reduction on the input signal, compressing it down into a smaller latent representation. The decoder then takes that latent representation and predicts what it expects the corresponding input to have been. This process of training the model to regenerate its input is a kind of unsupervised learning called representational learning. In the case of a denoising autoencoder, the model is instead given the noisy signal as input and the original clean audio signal is used to compute the loss function. This incentivizes the encoder to store only the information from the noisy signal that pertains to the clean audio and trains the decoder network to regenerate an estimate of the clean audio without the added noise. The resulting estimated clean audio is a purely synthetic generation produced by the model rather than a subtraction of noise from the input signal. The autoencoder architecture is not restricted, however, to a purely generative approach, as many SOTA models use a some variation of an encoder-decoder network to estimate spectrogram masks instead of the spectrogram directly.

### Sources for Autoencoders
I consulted a variety of sources to familiarize myself with autoencoders, but below are some that I found most helpful and from whom I drew the most inspiration:
* [Video Course]() by Valiero Valero, whose youtube channel is a treasure trove of content for learning machine learning audio.
* [Article](https://towardsdatascience.com/using-skip-connections-to-enhance-denoising-autoencoder-algorithms-849e049c0ac9) on using skip connections with autoencoders in keras. I derived my model architecture from this example.
* [Article](https://pyimagesearch.com/2020/02/24/denoising-autoencoders-with-keras-tensorflow-and-deep-learning/) on denoising images with autoencoders.

# Model Architecture

#### Encoder
* **input: spectrogram**
    * shape: (512, 256, 1)
* **7 convolutional layers**
    * 2^5, 2^6, 2^7, 2^8, 2^8, 2^9, 2^10 filters
    * 3x3 kernel size
    * 2x2 strides (downsampling)
* 1 flattened fully connected layer
    * 8192 neurons
* **output: 256 dimensional latent representation** (1d array of length 256)

#### Decoder (mirrors encoder)
* **input: latent representation**
    * shape: (256)
* **1 fully connected layer**
    * 8192 neurons
* **7 tansposed convolutional layers**
    * 2^10, 2^9, 2^8, 2^8, 2^7, 2^6, 2^5 filters
    * 3x3 kernel size
    * 2x2 strides (upsampling)
* **additive skip connections at 3rd and 5th convolution of encoder and decoder**
* **output: spectrogram**
    * shape: (512, 256, 1)

### Time - Frequency Modeling
The initial model was tra

### Phase Modeling

# Results

### Measuring Success
Working with audio data necessitates a combination of quantitative and qualitative (objective and subjective) assessment to measure the performance of a denoising model. The core metric that I used to train my model and compare the performance of different model architectures was the mean squared error of the predicted spectrogram of clean audio compared to the spectrogram of the actual original clean audio. While this measure is convenient in its ease of use and of interpretability, it suffers from a lack of phase-awareness [as described in this paper](https://openreview.net/pdf?id=SkeRTsAcYm). Since I planned to model phase in other ways, I decided that I would stick with a simpler loss term and error metric for this phase of the project. While it is difficult to translate mean squared error into the presence of audible artifacts, I am aiming for an mse below 0.001 on my testing data, as that predicted audio below that threshold has sounded acceptable to me.

In addition to measuring the error in spectrogram generation, I also listened through the predicted audio in the test dataset to get a feel for how the performance translated to the perceived sound itself.
### Latent Space Representations
* **mean**
* **std**
* **Note on Unused Latent Dimensions**
        
     Of the 256 latent dimensions 9 of them were left unused by the neural network, identified by both a mean and a standard deviation of 1 when measure across all latent representations created for the test set. This seems to be a fairly efficient use fo the latent space, as a reduction in dimensionality would typically cut the latent space in half to 128 dimensions. I would expect a signficant drop in performance to accompany a reduction in the size of the encoder bottleneck. Following the same logic, I would try increasing complexity in other areas of the architecture before expanding the size of the latent space, given that there is still headroom in the latent dimensions to store more information. 

# Next Steps

### Denoising Phase
The immediate next step building on the work already done within this project is a more thorough investigation of denoising phase with neural networks. Toward that end, I intend to:
1) experiment with using the real and complex components of the output of the fourier transform as direct inputs into an autoencoder architecture
2) further research the polar form of fourier transform outputs to better understand how I might more effectively use neural networks to estimate theta (the angle of the polar coordinate).

### Other Applications
In addition the use case of this model as a preprocessing component in an acoustic guitar tuner, two other applications have immediately come to mind as use cases for a similar denoising autoencoder:
1) An intelligent noise gate packaged as a VST plugin for use in a digital audio workstation or a digital soundboard for live audio.
2) Target specific speech enhancement for use in hearing aids. While the data collection requirements for such an implementation would be difficult, making costs potentially prohibitively high, in theory a similar architecture could be trained on a specific speaker's voice allowing for speaker-specific audio enhancement and custom remixing of audio levels for different settings. 

### Real-Time Processing
With the exception of the instrument-specific noise gate for digital audio workstations, all of the implementations of this model architecture would require that it be adapted and retrained for real-time inference on streaming audio from a microphone or some other recording source. While this is possible to do in python, a compiled language like C++ is generally the industry standard for real-time audio processing. I intend to begin exploring digital signal processing in C++ so that I can deploy this model in real-time applications. This, however, was outside of the scope of the project at hand.

# Data Sources and Licensing Information:
I compiled the dataset for this project from a variety of sources:

**Guitar Recordings**: I recorded the audio of the acoustic guitar tuning using logic pro for recording and editing, an Alesis V25 midi controller with the modulator wheel mapped to pitch shifting to immitate tuning a physical guitar, and the [Ample Sound AGT Plugin](https://www.amplesound.net/en/pro-pd.asp?id=6) as a playable midi acoustic guitar. This sample libray contains 5.21gb of recorded acoustic guitar samples with 9 different articulations recorded for each note on each string mapped across the midi velocity spectrum. The library also contains a variety of humanization attributes that are randomly applied to emulate human error and the associated noises when playing an acoustic guitar. The result is the industry standard for a playable midi acoustic guitar. I purchased a license to the Ample Sound AGT in 2021 for personal use, and decided that the convenience of the sample library and the highly realistic nature of the recordings made it a better choice than recording my own audio samples from scratch.

**Harmonic Noise**: When I refer to harmonic noise througout this project, I am referring to sound from other instruments or other pitched noise like a singing voice that may interfere with the proper function of a simple guitar tuner. The division betwee harmonic and environmental noise is not always a hard and fast distinction, as in some cases recordings classed as harmonic noise also contain noise from people talking such as in recordings of audio before the start of a concert or in a restaurant where music is playing. I sourced my harmonic noise files from logic pro stock audio loops and from a variety of live recording of instruments and other sound sources from [freesound.org](https://freesound.org). All recorded audio that I did not produce myself was used under a creative commons license. Detailed information about music used in this project and the respective license information is included [here](CREATIVE_COMMONS_ATTRIBUTION.md).

**Environmental Noise**: Environmental noise was sourced from the [ESC 50 Dataset](https://github.com/karolpiczak/ESC-50#license) and also used under a creative commons license. Specific license information can be found in the Creative Commons Attribution document listed above. This dataset was designed for environmental sound classification tasks, but I used it to add a random selection of non-harmonic noise signals to my audio signals to give more variation to the kinds of denoising tasks that the model would face. 