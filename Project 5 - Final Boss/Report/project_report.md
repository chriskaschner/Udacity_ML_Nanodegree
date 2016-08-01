# Capstone Project
## Machine Learning Engineer Nanodegree
Chris Kaschner
2016-08-1

## I. Definition
_(approx. 1-2 pages)_

### Project Overview

There is significantt value in a brand/ company/ organization being able to understand when and where it's products and services are being mentioned or discussed online.  Because of the popularity of services such as Instagram, Facebook, and Snapchat millions of photos [###todo factcheck and reference] are being uploaded on a daily bases to the internet.  Typically [###todo reference] these images are understood only as the text that is used to summarize/ comment on their contents, the image contents have, until recently, been of very little importance to what is understood about the post/ tweet/ message.  Instead only the tags or comments used to describe the image have informed other users or companies what is being displayed in an image.  

Companies already offer this type of structured data for text/ websites/ forum posts, https://www.diffbot.com/use-cases/#

What about images?  Is there a way to be able to identify the objects in an image to be able to know what brands are present in an image?  What if a customer had an issue and posted an image which alerted you to their plight and allowed you to communicate directly with them to resolve their problem?

My capstone project had the goal of identifying brands in untagged/ unlabeled photos from a social media feed.  Put another way, I wanted to answer the question "can you teach a computer to recognize brand logos in a feed of images from Instagram?".

I sought to identify 2 different shoe brands in a feed of images coming from Instagram.  [###todo### picture of Altra and Nike logos].  Importantly I also wanted to differentiate between the brands of interest and other shoes/ brands.  I.E. I didn't want to categorize everything as either Nike or Altra, but rather be able to know that an image did not contain one of the logos I was tracking.


*In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_*

### Problem Statement

Using machine learning tools can brands of interest be identified in unlabeled images?

IN order to solve this problem I undertake the following:

1. download existing pre-trained model and weights  [Keras](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) and [TensorFlow](https://www.tensorflow.org/versions/master/how_tos/image_retraining/index.html) both offer frameworks for creating the CNNs used, transferring weights, and popping layers to retrain.
3. remove/pop the final layers that are used to classify ImageNet images (1000 classes)
4. Replace final layers with ones that suit my needs, in this case 3 classes
5. Retrain the network on images that I provide
6. Measure performance and vary hyper parameters seeking to maximize the models performance
7. ...
8. Profit

The first building block used in my solution was a convolutional neural network and deep learning.

Convolutional neural networks are at the heart of these models. A way to reduce the dimensionality of the mathematics/ reduce computational complexity.  Likely to have overfitting in convolved features.  Pooling/ aggregating over a convolved feature, depends on mean/ max pooling

image layer -> convolved feature layer ->

There are a number of different types of neural networks, all loosely mimic the function of a brain, but again, that association is primarily in name.  Fundamentally they are complex systems that are built up from relatively simple constituents into a number of layers.

Of particular interest to this project is transfer learning.

Transfer learning is the ability to train a neural network in one scenario, say at Google in Mountain View, and then downloading the model and its associated weights to be reused in a different location/ time such as Austin, TX.

In the case of the inception v3 network would take about 2 weeks to train on 8x GPUs (~$3k Nvidia K40s).  Once trained the model and associated weights can be transferred to a smaller/ less expensive computer (in this case a MacbookPro) and used to classify images.

This approach allows us to "leverage" the resources/ computing power/ model complexity available to Google but used in our local environment.

The method I used to approach this problem utilized the significant work that large companies and research institutions have invested into this problem.  I used 2 different networks to attempt this, [VGG16](http://arxiv.org/pdf/1409.1556.pdf) and [Inception v3](http://arxiv.org/pdf/1512.00567v3.pdf).

The VGG16 model was created by the Visual Geometry Group (VGG) at Oxford University and was first described in arXiv:1409.1556. Although there are multiple versions, the one used in this work is 16-layers.

The Inception architecture [first appeared in a model called "GoogLeNet"](https://arxiv.org/abs/1409.4842) in 2014.  Version 3 [was described](http://arxiv.org/pdf/1512.00567v3.pdf) in 2015 and is the model used in this work.

Tools used to create neural networks

> [Keras](https://github.com/fchollet/keras) is a minimalist, highly modular neural networks library, written in Python and capable of running on top of eitherTensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

> [TensorFlow](https://github.com/tensorflow/tensorflow) is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code. TensorFlow also includes TensorBoard, a data visualization toolkit.
TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research


[###todo image of neuron and layers in a simplified net]

Explanation of a perceptron

Explanation of a network

Layers - > input, hidden, output layers

My intended solution to this problem will be as follows:

1. Identify a suitable framework/ library to build a pre-trained network (either Keras or TensorFlow)
2. Construct an existing network (either VGG 16 or Inception)
3. Load pre-trained weights into the previously built network, available [here for Keras](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) or here for TensorFlow [###todo links for weight files]
4. Remove the original model's top layers and replace with one suitable to perform our classification tasks
5. Update the bottlenecks for the

*In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
