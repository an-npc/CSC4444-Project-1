# Project 1: MNIST 
This project introduces you to both classical machine learning and deep learning approaches for image classification. You will work with the MNIST handwritten digit dataset, implement several classifiers, and compare their performance.

## Dataset: MNIST
~60,000 images

## Files on Moodle:
The raw MNIST dataset has been uploaded as image files (organized by digit).

Each student is required to create their own data partitions for the first two tasks (KNN and Naïve Bayes).

Example: randomly select 80% images for training and the rest for validation.

## Preprocessing:

Normalize pixel values to [0,1] (NumPy) or [-1,1] (PyTorch with transforms.Normalize).

Flatten into vectors (784 features) when needed.

Keep 2D shape for CNNs.

## Classifiers to Implement:
1. K-Nearest Neighbor (KNN) — NumPy only

    Load MNIST directly from the raw image files on Moodle.
    Create your own training and testing partitions.
    Represent each image as a flattened 784-dimensional vector.
    Compute Euclidean distance (required) and classify by majority vote among k neighbors.
    Test with at least k = 1, 3, 5.
    No scikit-learn KNeighborsClassifier.

2. Naïve Bayes — NumPy only

    Binarize each pixel (1 if pixel > 0.5, else 0).
    Estimate conditional probabilities of each pixel being “on” given a digit class.
    Apply Bayes’ rule with the independence assumption.

3. Linear Classifier — NumPy or PyTorch

    Implement a linear classifier: y = Wx.
    Train using L2 loss and gradient descent.
    Options:
        NumPy: write your own forward, backward, and update loop.
        PyTorch: use nn.Linear with optimizer and autograd.

4. Multilayer Perceptron (MLP) — PyTorch required

    At least one hidden layer with nonlinearity (e.g., ReLU).
    Example: 784 → 256 → 128 → 10.
    Train with SGD

5. Convolutional Neural Network (CNN) — PyTorch required

    At least two convolutional layers with activation + pooling.
    Example: Conv(1→32, 3×3) → ReLU → MaxPool → Conv(32→64, 3×3) → ReLU → MaxPool → FC → Softmax.
    Train with cross-entropy loss.
    Requirements:

## Deliverables
### Code:
    Modular, reproducible, and well-commented.
    Separate functions/files for each method.

### Report (3–5 pages):
    References: cite any sources used.
    AI Use Statement.
    Show your thinking process.

### Some interesting things to consider:
    Failure mode analysis (confusion matrix)
    Varying different hyperparameters.
    Try different loss functions.
    Show good visualization, like visualizing the weight matrix W of your linear classifier, or the probability map of your Naive Bayes as digit-like images.

## Tips
    Make use of AI tools. Example: ask ChatGPT “how should I load MNIST in PyTorch?”


# The Written Report (PDF file) 
submission must include the Title, Abstract, and the following sections:

## Project Description

Describe what your project is about: the problem your project/program addresses.
Motivations and potential applications of your project.

## AI Techniques

Briefly describe the main AI ideas/techniques behind it.

Example: if your project involves developing a smart agent for a board/video game, the main AI idea might be the agent learning evaluation functions from past experience.

## Datasets (if any)

Provide a brief description of the datasets used in your experiments.

## Implementation Tools

List the implementation platform, software tools, and packages used to develop your project.

## References and AI Assistance

Include references and note any AI help received during your work.
