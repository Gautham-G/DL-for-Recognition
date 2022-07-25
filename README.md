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
  

The main code/directory can be navigated as follows :

#### Code Structure

```console
.
├── src
│   └── vision
│       ├── __init__.py
│       ├── confusion_matrix.py (Functions to print and plot the confusion matrix)
│       ├── data_transforms.py (Functions to create different types of data augementation)
│       ├── dl_utils.py (Different utilities (loss, accuracy, save weights))
│       ├── image_loader.py (Dataloader)
│       ├── multilabel_resnet.py (Multi-label resnet model and forward)
│       ├── my_resnet.py (ResNet28 model and forward)
│       ├── optimizer.py (Helper functions for optimizer)
│       ├── runner.py (Runner - saves metadata)
│       ├── simple_net.py (Simple net as per desctription)
│       ├── simple_net_final.py (Model after transformations, as per desctiption)
│       └── stats_helper.py (stats helper)
```

Further, for results, please see the pdf : DL For Recog.pdf


### To Update : Instructions to use and download.
