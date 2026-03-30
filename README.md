# ML Samples

Machine learning and data science samples in Python — from classical algorithms to modern deep learning.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync
```

For specific sample areas with extra dependencies:

```bash
uv sync --extra whisper    # OpenAI Whisper transcription
uv sync --extra gcloud     # Google Cloud Vertex AI
uv sync --extra genai      # OpenAI API
```

## Samples

### Classical ML

| Directory | Description |
|-----------|-------------|
| [Linear_regression](Linear_regression/) | 1D linear regression |
| [Logistic_regression](Logistic_regression/) | Logistic regression (ecommerce, image classification) |
| [Multidimensional_regression](Multidimensional_regression/) | Polynomial fitting, L1/L2 regularization |
| [Decision_trees](Decision_trees/) | Decision trees, information gain, entropy |
| [Naive_bayes_mnist](Naive_bayes_mnist/) | Naive Bayes on MNIST |
| [Perceptron](Perceptron/) | Binary linear classifier on MNIST |
| [Regressions_sklearn](Regressions_sklearn/) | KNN and decision tree regression (scikit-learn) |

### Neural Networks

| Directory | Description |
|-----------|-------------|
| [Feedforward_neural_network](Feedforward_neural_network/) | Feedforward NN, backpropagation, softmax |
| [Optimizations](Optimizations/) | Momentum, RMSprop, Adam optimizers |
| [Other_samples](Other_samples/) | Gradient check, regularization, NN with 1 hidden layer |

### Deep Learning (TensorFlow / Keras)

| Directory | Description |
|-----------|-------------|
| [Tensorflow_basics](Tensorflow_basics/) | TF2 basics and sign language recognition app |
| [Tensorflow_distributed](Tensorflow_distributed/) | Distributed training with `tf.distribute.MirroredStrategy` |
| [tf2](tf2/) | TF2 ResNet on CIFAR-10 |
| [tf_book](tf_book/) | TensorFlow book samples |
| [Generative_DL](Generative_DL/) | Generative deep learning (CIFAR-10, Gemini, OpenAI) |

### Deep Learning (PyTorch)

| Directory | Description |
|-----------|-------------|
| [Pytorch_basics](Pytorch_basics/) | PyTorch fundamentals, 2-layer NN on MNIST |
| [Pytorch](Pytorch/) | PyTorch CUDA tensor operations |
| [Convolutional_Neural_Networks](Convolutional_Neural_Networks/) | CNN samples (PyTorch, TF2 ResNet) |

### Tools and Integrations

| Directory | Description |
|-----------|-------------|
| [whisper](whisper/) | OpenAI Whisper audio transcription with Minio upload |
| [gcloud](gcloud/) | Google Cloud Vertex AI function calling |
| [Tools](Tools/) | Model converter utilities |
| [learn](learn/) | Async Python examples |
