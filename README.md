# AudioVerify
A Repo to try out some Voice verification models on Android


## Getting Started
### Import the project (manual)
1. Clone the project into a folder on your computer
1. Import this project by clicking File > New > Import Project from the Android Studio menu

### Import the project (VCS)
To import the project via Android Studio VCS:
1. Click on File > New > Project from Version Control > Git
1. Copy-Paste the git url to this repo into the field on the dialog and click 'Clone'

### Download models
The models referenced in the porject had to be hosted in this [Google drive folder](https://drive.google.com/drive/folders/1Lrx-im_AUXSA5KAf14MSI7zMAEig0V1U?usp=sharing) because of their sizes.
[Click here](https://drive.google.com/drive/folders/1Lrx-im_AUXSA5KAf14MSI7zMAEig0V1U?usp=sharing) to download the models and save them in the project's ```assets``` folder

### Build Project
Build and run the project, resolving any environment specific errors.

## Testing out the implementations
After a successful build and run, you will be able to click couple of buttons and watch the AS logs. Currently, only one of the four approaches to loading the models is able to load a model correctly
- **Load .h5 model with Deeplearning4J**: This attempts to load a raw Keras (.h5) model using Deeplearning4J library. This library does not build as there are Class conflicts in the library.
This could have been the best approach as far as performance is concerned as the original weights of the model is presserved. This library is curently commented out in ```build.gradle```
- **Load .tflite model with Firebase**: Here we try to load a TensorFlow Lite (.tflite) model using Firebase ML library. Model loads without issues. But now how to get a 4D ```float32``` array from a wav file to feed into model??.
Currently, converting a wav file into a 4D ```float32``` array has not been figured out yet.
- **Load .tflite model with TensorFlow**: This approach loads a Tensorflow (.tflite) model using TensorFlow library. This approach crashes the app with a FlatBuffer error
- **Load .pt model with Pytorch**: In this approach, we try to load Pytorch model using Pytorch library. This approach also crashes the app with invalid model error

## Next steps
- Figure out how to convert an audio file into a 4D ```float32``` array in java (Android) to feed into the Firebase loaded model
- Maybe also do something about the file size of the model
