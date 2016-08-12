# Project 5 - Brand Recognition

An [estimated 2 trillion photos](http://ben-evans.com/benedictevans/2015/8/19/how-many-pictures) were shared online in 2015, with that number expected to continue to grow.  Lacking efficient methods to examine and identify objects in these images they can only be understood by the text that is used to summarize or tag them and not the actual image content.

Is there a way to be able to identify the objects in an image to be able to know what brands are present in an image?  This type of structured data is already [available for text](https://www.diffbot.com/) but what about images?  What if a customer had an issue and posted an image without your company name in the text description?  How would you could you identify and find such images?

In this project I create a Convolutional Neural Network (CNN) that is capable of identifying brands in untagged/ unlabeled photos from a social media feed.  The model I use implements a [previously trained](https://github.com/tensorflow/models/tree/master/inception) network and [transfer learning](https://en.wikipedia.org/wiki/Inductive_transfer) to speed training.

In this project I build a convolutional neural network to classify images based on the appearance of brand logos.  

To do that I:

1. Construct the Inception v3 network in [TensorFlow](https://www.tensorflow.org/)
1. Load pre-trained weights into the previously built network
1. Modify the pre-existing model for our classification tasks
1. Update bottlenecks
1. Retrain the network on images of interest
1. Record & Evaluate model performance
1. Visualize Outputs using [Keras](https://keras.io/)

This represent the capstone project for the Udacity Machine Learning Nanodegree program.

# Requirements:
This project requires **Python 2.7**

### Visualizng VGG16 CNN Layers
Use the Jupyter notebook `VisualizeLayersVGG16.ipynb`, fill in the paths as commented, and follow the [the process described here](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html).

### Running TensorFlow
In order to reconstruct these results locally, follow the TensorFlow [tutorial](https://www.tensorflow.org/versions/master/how_tos/image_retraining/index.html) on transfer learning and retraining.  Once you have that setup, replace the existing `retrain.py` file with the [version here](https://github.com/chriskaschner/Udacity_ML_Nanodegree/tree/master/Project%205%20-%20Brand%20Recognition/image_retraining) and the dataset available [here](https://dl.dropboxusercontent.com/u/969119/DatasetForMLN.zip).

For the ideal binary classification model, run from the command line with the following command
`python retrain.py --image_dir [replace with img directory] --output_graph [directory]/output_graph.pb --output_labels [directory]/output_labels.txt --bottleneck_dir [directory]/bottlenecks --summaries_dir [directory]/logs/ --how_many_training_steps 8000 --learning_rate 0.001`

For the ideal multiclass classification model, run from the command line with the following command
`python retrain.py --image_dir [replace with img directory] --output_graph [directory]/output_graph.pb --output_labels [directory]/output_labels.txt --bottleneck_dir /[directory]/bottlnecks --summaries_dir [directory]/logs --how_many_training_steps 8000 -learning_rate 0.001 --adagrad_accumulator 0.01 --optimizer Adagrad`

### Making Predicitons with a TensorFlow Model
Use the Jupyter notebook `TensorFlowInceptionClassification.ipynb` and fill in the paths as commented.
