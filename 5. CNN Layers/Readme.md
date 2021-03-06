## My Learnings in this lesson

This Readme contains all basic terminologies that I have come across during this lesson.

----

### Keywords
    Convolution Neural Networks
    Fashion MNIST
    Layers - Conv, Pooling, Dropout, FC, Softmax
    SGD using Momentum
    Feature Visualization - Nearest Neighbours, Dimensionality Reduction
    Pytorch Documentation

---
### Data Preprocessing

![Mnist Preprocessing](images/mnist_preprocessing.png "Preprocessing steps for Fashion MNIST dataset creation")

### CNN's
For filters and their uses, see the video *Convolution Layers* in the folder.

### CNN Structure
A classification CNN takes in an input image and outputs a distribution of class scores, from which we can find the most likely class for a given image.

![CNN Structure](images/cnn_structure.png "A classification CNN Structure")

----

### CNN Layers
The CNN itself is comprised of a number of layers; layers that extract features from input images, reduce the dimensionality of the input, and eventually produce class scores

![CNN Detailed Structure](images/cnn_detailed.png "A detailed classification CNN Structure")


#### Pooling Layers
After a couple of convolutional layers (+ReLu's), you'll see a maxpooling layer.

- Pooling layers take in an image (usually a filtered image) and output a reduced version of that image
-  Pooling layers reduce the dimensionality of an input
- Maxpooling layers look at areas in an input image and choose to keep the maximum pixel value in that area, in a new, reduced-size area.
- Maxpooling is the most common type of pooling layer in CNN's, but there are also other types such as average pooling.

#### Fully-Connected Layer
- A fully-connected layer's job is to connect the input it sees to a desired form of output
- Typically, this means converting a matrix of image features into a feature vector whose dimensions are 1xC, where C is the number of classes. 
- As an example, say we are sorting images into ten classes, you could give a fully-connected layer a set of [pooled, activated] feature maps as input and tell it to use a combination of these features (multiplying them, adding them, combining them, etc.) to output a 10-item long feature vector. This vector compresses the information from the feature maps into a single feature vector.

#### Softmax
- The very last layer you see in networks is a softmax function. 
- The softmax function, can take any vector of values as input and returns a vector of the same length whose values are all in the range (0, 1) and, together, these values will add up to 1. 
- This function is often seen in classification models that have to turn a feature vector into a probability distribution.

- The fully-connected layer can turn feature maps into a single feature vector that has dimensions 1xn. the softmax function turns that vector into a n-item long probability distribution in which each number in the resulting vector represents the probability that a given input image falls in class 1, class 2, class 3, ... class n.

#### Dropout
- Convolutional, pooling, and fully-connected layers are all you need to construct a complete CNN, but there are additional layers that you can add to avoid overfitting, too.
- Dropout layers essentially turn off certain nodes in a layer with some probability, p.
- This ensures that all nodes get an equal chance to try and classify different images during training, and it reduces the likelihood that only a few, heavily-weighted nodes will dominate the process.

----
#### Momentum 
Refer video in videos/ folder

----

### Feature Vizualization
- The first layer usually learns smaller feature maps such as edges, color gradients, etc. directly from the input image. Usually, it represents th high-pass filter learning phase of the model.
- Since the weights are just another sets of matrices, for the first Layer, we can directly visualze these weights to understand what this layer has learned.
- Since the deeper layers are not directly connected to the input image, we cannot directly visualize what they are learning using the same technique as the First Conv Layer.
*See the feature visualization notebook for more info*

#### Visualizing the final feature vector
- To visualize what a vector represents about an image, we can compare it to other feature vectors, produced by the same CNN as it sees different input images. We can run a bunch of different images through a CNN and record the last feature vector for each image. This creates a feature space, where we can compare how similar these vectors are to one another.
- We can measure vector-closeness by looking at the **nearest neighbors** in feature space. Nearest neighbors for an image is just an image that is near to it; that matches its pixels values as closely as possible. So, an image of an orange basketball will closely match other orange basketballs or even other orange, round shapes like an orange fruit.
- **Nearest neighbors in feature space**: 
In feature space, the nearest neighbors for a given feature vector are the vectors that most closely match that one; we typically compare these with a metric like MSE or L1 distance. And these images may or may not have similar pixels, which the nearest-neighbor pixel images do; instead they have very similar content, which the feature vector has distilled.
- **Dimensionality reduction**:
Another method for visualizing this last layer in a CNN is to reduce the dimensionality of the final feature vector so that we can display it in 2D or 3D space.
    - *Principal Component Analysis*
    One is PCA, principal component analysis, which takes a high dimensional vector and compresses it down to two dimensions. It does this by looking at the feature space and creating two variables (x, y) that are functions of these features; these two variables want to be as different as possible, which means that the produced x and y end up separating the original feature data distribution by as large a margin as possible.
    - *t-SNE*:
    TEE-SNEE tands for t-distributed stochastic neighbor embeddings. It’s a non-linear dimensionality reduction that, again, aims to separate data in a way that clusters similar data close together and separates differing data.

---

### Some Useful Links
- [Cezannec's Blog ](https://cezannec.github.io/Convolutional_Neural_Networks/)

- [Pytorch NN Documentation](https://pytorch.org/docs/stable/nn.html)

- [Pytorch Loss Function](https://pytorch.org/docs/master/nn.html#loss-functions)