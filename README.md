# VisConvNets
This is a personal project for the purpose of understanding Convolution networks with visualizations of intermediate Convolution and Activation layers. 

I have used the following dataset https://www.kaggle.com/alxmamaev/flowers-recognition for training my InceptionV3 model.

You can use create_train_val_test.py to create Training, Validation and Test directories out of original datasets. After that you can just run model_training_and_visualization.py to train the model and produce all the visualizations shown in my blog post. Note that the images might differ depending on what test image you are using and how your network is trained.

Although I have tried to explain my code with comments, but for any issues you can just create a Issue and I'll try to resolve it. I have taken the code inspiration from the book Deep Learning in Python by Francois Chollet. I highly recommend for everybody to read this book espeicially if you are new to the field.

-----------------------------------------------------------------------------------------------------------

## Concept behind the project
From the Face Recognition in your phone to driving your cars, the tremendous power of CNNs is being used to solve many real-world problems.

But despite the wide availability of large databases and pre-trained CNN models, sometimes it becomes quite difficult to understand what and how exactly your large model is learning, especially for people without the required background of Machine Learning.

The goal of most people is to just use the pre-trained model to some image classification or any other related problem to get the final results. They are least bothered about the internal workings of the network, which can actually tell them a lot about how and what their network is learning and also debug its failures.

In this project, I have summarized the three techniques that I learned and my results to further understanding the internal working on convnets.

## Enter Visualizations
Visualizing the output of your machine learning model is a great way to see how its progressing, be it a tree-based model or a large neural network. While training deep networks, most people are only concerned with the training error(accuracy) and validation error(accuracy). While judging these two factors does give us an idea of how our network is performing at each epoch, when it comes to deep CNN networks like Inception there is so much more that we can visualize and thus learn about network architecture.

I will demonstrate few ways of visualizing your model outputs (intermediate as well as final layers), which can help you gain more insight into working of your model. I trained the InceptionV3 model (pre-trained on ImageNet) available in Keras, on Flower recognition dataset available on Kaggle.

I trained the model for 10 epochs with a batch size of 32, with each image resized to a shape of (299, 299, 3), which is required by the pre-trained InceptionV3 model. My model was able to reach training loss of 0.3195 and validation loss of 0.6377. I used Keras inbuilt ImageDataGenerator module for augmenting images, so that the model does not over-fit too quickly.

## Visualizing Intermediate Layer Activations
For understanding how our deep CNN model is able to classify the input image, we need to understand how our model sees the input image by looking at the output of its intermediate layers. By doing so, we are able to learn more about the working of these layers.

For instance, following are the outputs of some of the intermediate convolution and their corresponding activation layers of the trained InceptionV3 model, when provided with image of a flower from test set.

> __*Figure: Original Image*__
![Original Image](https://github.com/ajaypt92/VisConvNets/blob/master/Images/OriginalImage.jpeg)

> __*Figure: Conv2d_1*__
![Conv2d_1](https://github.com/ajaypt92/VisConvNets/blob/master/Images/conv2d_1_grid.jpg)

