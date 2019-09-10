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

> __*Figure: Conv2d_4*__
![Conv2d_4](https://github.com/ajaypt92/VisConvNets/blob/master/Images/conv2d_4_grid.jpg)

> __*Figure: Conv2d_9*__
![Conv2d_9](https://github.com/ajaypt92/VisConvNets/blob/master/Images/conv2d_9_grid.jpg)

> __*Filters from layers First, Fourth and Ninth convolution layers in InceptionV3*__

> __*Figure: activation_1*__
![activation_1](https://github.com/ajaypt92/VisConvNets/blob/master/Images/activation_1_grid.jpg)

> __*Figure: activation_4*__
![activation_4](https://github.com/ajaypt92/VisConvNets/blob/master/Images/activation_4_grid.jpg)

> __*Figure: activation_9*__
![activation_9](https://github.com/ajaypt92/VisConvNets/blob/master/Images/activation_9_grid.jpg)

> __*Filters from ReLU activation layers respective to First, Fourth and Ninth convolution layers in InceptionV3*__

The above figures show the filters from few intermediate convolution and ReLU layers respectively from InceptionV3 network. I captured these images by running the trained model on one of the test images.

If you take a look at the different images from Convolution layers filters, it is pretty clear to see how different filters in different layers are trying to highlight or activate different parts of the image. Some filters are acting as edge detectors, others are detecting a particular region of the flower like its central portion and still others are acting as background detectors. Its easier to see this behavior of convolution layers in starting layers, because as you go deeper the pattern captured by the convolution kernel become more and more sparse, so it might be the case that such patterns might not even exist in your image and hence it would not be captured.

Coming to the ReLU (Rectified Linear Units) activations of corresponding convolution layers, all they do is apply the Relu function to each pixel which is ReLU(z) = max(0, z), as shown in below figure . So, basically, at each pixel the activation function just puts either a 0 for all negative values or pixel value itself if it is greater than 0.

> __*Figure: ReLU function*__
![ReLU function](https://github.com/ajaypt92/VisConvNets/blob/master/Images/ReLU.png)

By visualizing the output from different convolution layers in this manner, the most crucial thing that you will notice is that the layers that are deeper in the network visualize more training data specific features, while the earlier layers tend to visualize general patterns like edges, texture, background etc. This knowledge is very important when you use Transfer Learning whereby you train some part of a pre-trained network (pre-trained on a different dataset, like ImageNet in this case) on a completely different dataset. The general idea is to freeze the weights of earlier layers, because they will anyways learn the general features, and to only train the weights of deeper layers because these are the layers which are actually recognizing your objects.

## Visualizing Convnet Filters
Another way of learning about what your Convolution network is looking for in the images is by visualizing the convolution layer filters. By displaying the network layer filters you can learn about the pattern to which each filter will respond to. This can be done by running Gradient Descent on the value of a convnet so as to maximize the response of a specific filter, starting from a blank input image.

Here are some of the patterns from the InceptionV3 model that I trained on Flowers dataset.

> __*Figure: Filters from third convolution layer in InceptionV3*__
![Filters from third convolution layer in InceptionV3](https://github.com/ajaypt92/VisConvNets/blob/master/Images/conv2d_3_filters.jpg)

> __*Figure: Filters from eighth convolution layer in InceptionV3*__
![Filters from eighth convolution layer in InceptionV3](https://github.com/ajaypt92/VisConvNets/blob/master/Images/conv2d_8_filters.jpg)

> __*Figure: Filters from fortieth convolution layer in InceptionV3*__
![Filters from fortieth convolution layer in InceptionV3](https://github.com/ajaypt92/VisConvNets/blob/master/Images/conv2d_40_filters.jpg)

After taking a closer look at these images of filters from different convolution layers, it becomes very clear what different layers are actually trying to learn from the image data provided to them. The patterns found in filters in starting layers seem to be very basic, composed of lines and other basic shapes, which tell us that the earlier layers learn about basic features in images like edges, colors, etc. But as you move deeper into the network, the patterns get more complex, suggesting that the deeper layers are actually learning about much more abstract information, which helps these layers to generalize about the classes and not the specific image. And this is why, we saw a few empty filters activations in deeper layers in the previous section, because that particular filter was not activated for that image, in other words, the image does not have the information that the filter was interested in.

## Visualizing Heatmaps of class activations
While predicting the class labels for images, sometimes your model will predict wrong label for your class, i.e. the probability of the right label will not be maximum. In cases such as these, it will be helpful if you could visualize which parts of the image is your convnet looking at and deducing the class labels.

The general category of such techniques is called Class Activation Map (CAM) visualization. One of the techniques of using CAM is to producing heatmaps of class activations over input images. A class activation heatmap is a 2D grid of scores associated with a particular output class, computed for every location for an input image, indicating how important is each location is with respect to that output class.

> __*Figure: Class Activation Map (CAM) visualization_1*__
![Class Activation Map (CAM) visualization_1](https://github.com/ajaypt92/VisConvNets/blob/master/Images/CAM_1.jpeg)

> __*Figure: Class Activation Map (CAM) visualization_2*__
![Class Activation Map (CAM) visualization_2](https://github.com/ajaypt92/VisConvNets/blob/master/Images/CAM_2.jpeg)

> __*Figure: Class Activation Map (CAM) visualization_3*__
![Class Activation Map (CAM) visualization_3](https://github.com/ajaypt92/VisConvNets/blob/master/Images/CAM_3.jpeg)

In the above images, you can see how this technique works. Starting from left, first is the input image, then comes the activation heatmap of the last Mixed layer in the InceptionV3 architecture, and finally I have superimposed the heatmap over the input image. So basically, what the heatmap is trying to tell us is the locations in the image which are important for that particular layer to classify it as the target class, which is Daisy in this case.

In the first image, it is pretty clear that the network has no problem in classifying the flower, as there are no other objects in the entire image. In the next image, the network could not classify the image as Daisy, but if you take a look at the heatmap of the activation map, it is clear that the network is looking for the flowers in the correct parts of image. Similar is the case for the third image, the network is able to highlight the bottom left part of the image, where small daisy flowers are located. So its pretty clear that while the network is not able to correctly classify second and third images correctly, but the reason for this decision is not the incorrectness of network, but rather the larger fraction of image occupied by other objects in both images.

The activation heatmaps may differ for different layers in the network, as all layers view the input image differently, creating a unique abstraction of image based on their filters. In this example, I have focused on the final layer of model, as the class prediction label will largely depend on it. But it can be a good experiment to compare the activation heatmaps of different layers.

## Conclusion
In this project, I have described three different methods for visualizing your deep convolution network. As explained earlier, visualizations like these can help us understand the working of black-box techniques, like neural networks better, which can be useful for debugging any errors or the performance of network in totality.
