# Recognition with Deep Learning

**Overview** : This project uses Deep Convolutional Neural Networks for scene recognition. In essence, the motivation of the project is to study how various transformations can affect performance of our created models incrementally. 

To summarize the project, please see below :

* First, I start out with a simple CNN built from scratch which is aptly named *SimpleNet*.
* Next, to increase prediction accuracy, I implement a few 'tricks' which is show as below. The model for this is named *SimpleNetFinal*.
  *  Data augmentation techniques
  *  Zero Centering and Normalizing Variance of the image
  *  Using Dropout regularization
  *  Increasing the complexity of the network by applying extra layers of conv & Relu
  *  And lastly, to counter the increase in training time and the 'brittleness' of the model from the previous steps, batch normalization is applied.
* Lastly, I fine tune a pre-trained composite CNN architecture : ResNet. The objective is to fine-tune the ResNet architecture to recognize scenes, which it was not initially trained to do.
* A multi-label ResNet is later created to predict the multiple attributes (like clouds, water body, people, nature etc.) in an image.
  

The code directory can be found as follows :

## Project Structure

```console
.
├── README.md
├── cutoff_frequencies.txt
├── data
│   ├── 1a_dog.bmp
│   ├── 1b_cat.bmp
│   ├── 2a_motorcycle.bmp
│   ├── 2b_bicycle.bmp
│   ├── 3a_plane.bmp
│   ├── 3b_bird.bmp
│   ├── 4a_einstein.bmp
│   ├── 4b_marilyn.bmp
│   ├── 5a_submarine.bmp
│   └── 5b_fish.bmp
│   └── part4
│       ├── kernel.npy
│       └── mystery.npy
├── docs
│   └── report.pptx
├── project-1.ipynb
├── pyproject.toml
├── scripts
│   └── submission.py
├── setup.cfg
├── src
│   └── vision
│       ├── __init__.py
│       ├── part1.py
│       ├── part2_datasets.py
│       ├── part2_models.py
│       ├── part3.py
│       └── utils.py
└── tests
'''



### To Update : Instructions to use and download.
