# Image Retraining

Files for duplicating my final project.

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

[### links to tutorials]
