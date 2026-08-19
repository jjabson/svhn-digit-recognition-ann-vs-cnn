#show figure.where(kind: table): set figure.caption(position: top)

#import "../generated/report_data.typ": *

#set page(
  paper: "us-letter",
  margin: (
    top: 0.8in,
    bottom: 0.8in,
    left: 0.85in,
    right: 0.85in,
  ),
)

#set text(
  font: "Libertinus Serif",
  size: 10.5pt,
)

#set par(
  justify: true,
  leading: 0.65em,
)

#set heading(numbering: "1.")

#align(center)[
  #v(1.5in)

  #text(size: 24pt, weight: "bold")[
    SVHN Digit Recognition
  ]

  #v(0.25in)

  #text(size: 15pt)[
    CNN Training, Evaluation, API Deployment, and Dockerization
  ]

  #v(0.8in)

  #text(size: 12pt)[
    Machine Learning Engineering Project Report
  ]

  #v(1.1in)

  #text(size: 11pt)[
    Author:
    Jerome Jabson
  ]

  #v(0.15in)

  #text(size: 11pt)[
    Version: 1.0
  ]

  #v(0.15in)

  #text(size: 11pt)[
    Date:
    July 2026
  ]
]

#pagebreak()

#outline(
  title: [Table of Contents],
  indent: auto,
)

#pagebreak()

#outline(
  title: [List of Figures],
  target: figure.where(kind: image),
)

#pagebreak()

#outline(
  title: [List of Tables],
  target: figure.where(kind: table),
)

#pagebreak()

= Executive Summary

This report documents the development and deployment of a convolutional
neural network for classifying cropped street-view house-number images from
the SVHN dataset.

The project includes data exploration, preprocessing, CNN model development,
model evaluation, modular inference code, a FastAPI prediction service, and
Docker-based deployment. The completed API accepts an uploaded digit image
and returns the predicted class, numeric confidence, human-readable confidence
percentage, and the probability assigned to each digit class.

= Problem Statement

Recognizing handwritten or street-view digits is a fundamental computer vision
task with applications in mail sorting, package delivery, navigation systems,
utility meter reading, traffic monitoring, and autonomous vehicles. While the
problem appears simple to humans, variations in lighting, viewing angles,
background clutter, and digit shape make automated recognition challenging.

The Street View House Numbers (SVHN) dataset provides real-world images of
house numbers captured from Google Street View. Unlike handwritten digit
datasets such as MNIST, SVHN contains naturally occurring images with greater
variation in color, scale, orientation, and background complexity. These
characteristics make SVHN a more realistic benchmark for evaluating modern
computer vision models.

The objective of this project is to design, train, evaluate, and deploy a
Convolutional Neural Network (CNN) capable of accurately classifying cropped
house-number images into one of the ten digit classes (0–9). Beyond model
development, the project demonstrates production-oriented machine learning
engineering practices by exposing the trained model through a FastAPI REST API
and packaging the complete application in a Docker container for reproducible
deployment.

= Project Objectives

The primary objectives of this project were to design, implement, evaluate, and
deploy a deep learning solution for handwritten digit recognition using the
Street View House Numbers (SVHN) dataset.

The project objectives are summarized below:

- Develop a Convolutional Neural Network (CNN) capable of classifying digits
  into one of ten classes (0–9).

- Perform data preprocessing, normalization, and label preparation suitable
  for deep learning.

- Evaluate model performance using accuracy, precision, recall, F1-score,
  and confusion matrices.

- Separate model inference from training by implementing a reusable inference
  module.

- Develop a REST API using FastAPI that accepts uploaded images and returns
  prediction results in JSON format.

- Containerize the application using Docker to ensure reproducible deployment
  across different computing environments.

- Validate the complete prediction pipeline using representative sample images
  from the test dataset.

  = Dataset Description

This project uses the Street View House Numbers (SVHN) dataset, one of the most
widely used benchmark datasets for image classification. The dataset consists
of cropped color images extracted from Google Street View photographs, where
each image contains a single digit.

Unlike handwritten digit datasets such as MNIST, the SVHN dataset contains
naturally occurring images captured in outdoor environments. Variations in
lighting, background objects, viewing angle, digit size, and image quality make
the classification task significantly more challenging and representative of
real-world computer vision applications.

Each image was converted to grayscale during preprocessing, producing a
32 × 32 pixel input image suitable for training the convolutional neural
network.

@dataset-split summarizes the number of images allocated to the
training, validation, and testing datasets.

#figure(
  table(
    columns: (1fr, 1fr),
    align: (left, center),
    inset: 6pt,

    table.header(
      [*Dataset Split*],
      [*Images*],
    ),

    [Training],   [#training-images],
    [Validation], [#validation-images],
    [Testing],    [#testing-images],
  ),
  caption: [Dataset Split Summary],
) <dataset-split>

@dataset-characteristics summarizes the primary characteristics
of the dataset used in this project.

#figure(
  table(
    columns: (1fr, 1.5fr),
    align: (left, center),
    inset: 6pt,

    table.header(
      [*Property*],
      [*Value*],
    ),

    [Image Size],    [#image-dimensions],
    [Channels],      [#image-channels],
    [Classes],       [#class-summary],
    [Data Type],     [#data-type],
    [Learning Task], [#learning-task],
  ),
  caption: [SVHN Dataset Characteristics],
) <dataset-characteristics>

= Exploratory Data Analysis

Before training the convolutional neural network, the dataset was explored to
better understand its composition, class distribution, and visual
characteristics. Exploratory Data Analysis (EDA) helps verify data quality,
identify potential class imbalance, and ensure that preprocessing decisions are
appropriate before model development.

The analysis focused on answering the following questions:

- What do the digit images look like?
- Is the dataset balanced across all digit classes?
- Are there any obvious anomalies or quality issues?
- What preprocessing steps are required before training?

== Representative Digit Images

@sample-digits presents one representative image from each of the ten
digit classes. The images demonstrate the variation in brightness, contrast,
stroke thickness, and visual appearance found in the dataset.

#figure(
  image("../figures/figure1_sample_digits.png", width: 92%),
  caption: [
    Representative images for the ten digit classes in the SVHN dataset.
  ],
) <sample-digits>

As shown in @sample-digits, the images contain naturally occurring
variations that make the classification problem more challenging than
recognizing clean, standardized handwritten digits.

== Class Distribution

A balanced class distribution is important because it reduces the risk that the
model will favor digits that appear more frequently during training.
@class-distribution shows the number of training images available for each digit
class.

#figure(
  image("../figures/figure2_class_distribution.png", width: 90%),
  caption: [
    Distribution of training images across the ten SVHN digit classes.
  ],
) <class-distribution>

@class-distribution shows that the training dataset is well balanced. Each digit
class contains approximately #average-class-count images, with only minor variation between
the least represented and most represented classes. The smallest class contains
#smallest-class-count images, while the largest contains #largest-class-count images, a
difference of only #class-count-difference samples.

Because no class is substantially underrepresented, class weighting or
oversampling was not required for model training. This balanced distribution
also makes overall accuracy a more informative evaluation metric, although
precision, recall, F1-score, and the confusion matrix are still used to assess
class-level performance.

== Key EDA Observations

The exploratory analysis produced the following findings:

- The dataset contains ten clearly defined target classes representing the
  digits 0 through 9.

- All images have consistent dimensions of 32 × 32 pixels, which simplifies
  batching and model input preparation.

- The training data is evenly distributed across the ten digit classes, so no
  major class-imbalance treatment was necessary.

- The images contain realistic variation in brightness, contrast, stroke width,
  orientation, and surrounding visual context.

- Some images contain portions of neighboring digits or background details,
  making the task more challenging than classification using clean handwritten
  digit datasets.

- Pixel-value normalization is required before training so that the neural
  network receives inputs on a consistent numerical scale.

 = Data Preprocessing

The raw SVHN arrays were transformed into a representation suitable for
convolutional neural-network training. The preprocessing workflow included
reshaping the image arrays, normalizing pixel values, and encoding the target
labels.

== CNN Input Shape

The original grayscale images were stored as two-dimensional arrays with
dimensions #image-dimensions. Prior to training, a channel dimension was added
to each image so that every sample had the shape *(32, 32, 1)*, matching the
input format expected by the convolutional neural network.

The added final dimension represents the single grayscale channel. As a result,
the complete training dataset is represented as a four-dimensional tensor with
the shape *#training-tensor-shape*.

== Pixel-Value Normalization

Neural networks generally train more reliably when input values are placed on a
consistent numerical scale. The original image pixels were represented using
values in the range #original-pixel-range. Each pixel was divided by
#normalization-divisor so that the resulting values fell within the range
#normalized-pixel-range.

@normalization-comparison compares the same image before and after
normalization. Although the two images appear visually identical, the numerical
representation of every pixel has changed.

#figure(
  image(
    "../figures/figure3_normalization_comparison.png",
    width: 92%,
  ),
  caption: [
    Comparison of an SVHN image before and after pixel-value normalization.
  ],
) <normalization-comparison>

The original image uses pixel values in the range
#original-pixel-range, while the normalized image uses values in the range
#normalized-pixel-range. Normalizing the input improves numerical stability
during gradient-based optimization while preserving the visual information
contained in the image.

== Pixel-Value Distribution

While @normalization-comparison illustrates the effect of normalization on a
single image, it does not show how the transformation affects the dataset as a
whole. To better understand the preprocessing step, the distribution of pixel
values was analyzed before and after normalization.

#figure(
  image(
    "../figures/figure4_pixel_distribution.png",
    width: 100%,
  ),
  caption: [
    Distribution of pixel values before and after normalization across the
    complete SVHN training dataset.
  ],
) <pixel-distribution>

@pixel-distribution demonstrates that normalization preserves the overall shape
of the pixel-value distribution. The transformation changes only the numerical
scale, mapping the original range of #original-pixel-range to
#normalized-pixel-range through a linear scaling operation.

The distribution is concentrated around mid-range pixel intensities rather than
being uniformly distributed across the available range. Because normalization is
a linear transformation, the relative frequency of each pixel intensity is
preserved while producing values that are more suitable for gradient-based
optimization during model training.

== Target Encoding

The image preprocessing steps described in the previous sections prepared the
input data for the convolutional neural network. The target labels also required
preprocessing before model training could begin.

Each training image is associated with one of the ten digit classes (0 through
9). Although these labels are stored as integer values, the CNN produces a
probability for every output class rather than a single integer prediction.
Consequently, the labels were converted into one-hot encoded vectors containing
#one-hot-vector-length positions.

For example, the digit label 3 becomes:

#block(
  inset: 8pt,
  radius: 4pt,
  stroke: luma(220),
  fill: luma(248),
)[
`[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
]

Each position in the vector corresponds to one output neuron in the softmax
classification layer. The correct class is represented by a value of 1, while
all remaining positions contain 0.

This representation enables the network to learn a probability distribution over
all digit classes while using categorical cross-entropy as the training loss
function.

== Training–Serving Consistency

The preprocessing techniques described throughout this chapter prepare the
training data for the convolutional neural network. These same transformations
must also be applied during deployment so that the model receives data in the
same format used during training.

For every image submitted to the FastAPI prediction endpoint, the preprocessing
pipeline performs the following operations:

- Convert the image to grayscale.
- Resize the image to #image-dimensions.
- Normalize pixel values from #original-pixel-range to
  #normalized-pixel-range.
- Add the single-channel dimension required by the CNN.
- Reshape the image into the batch format expected by the model.

By sharing identical preprocessing logic between the training pipeline and the
deployment pipeline, the project avoids *training-serving skew*, a condition in
which a deployed model receives data that has been prepared differently from the
data used during training.

Maintaining a consistent preprocessing pipeline ensures that every prediction is
generated from images with the expected dimensions, numerical scale, channel
configuration, and tensor structure. This consistency improves reliability and
helps the deployed model reproduce the performance observed during model
evaluation.

= CNN Architecture

The convolutional neural network was designed to progressively transform the
preprocessed grayscale input images into increasingly abstract visual features
before producing a probability distribution across the ten digit classes.

The architecture consists of two feature-extraction blocks followed by a
fully connected classification head. The convolutional blocks learn spatial
patterns from the input images, while the classification head converts the
learned feature representation into the final digit prediction.

== High-Level Architecture

@cnn-architecture summarizes the major stages of the trained convolutional
neural network. The model accepts a single grayscale image with dimensions
#model-input-shape and passes it through two successive feature-extraction blocks
before reaching the classification head.

#figure(
  image(
    "../figures/figure5_cnn_architecture.png",
    width: 90%,
  ),
  caption: [
    High-level architecture of the trained SVHN convolutional neural network.
  ],
) <cnn-architecture>

The first feature-extraction block operates on the original 32 × 32 spatial
representation and produces 32 feature maps. A max-pooling operation then
reduces the spatial dimensions to 16 × 16.

The second feature-extraction block increases the learned representation to
64 feature maps while a second pooling operation reduces the spatial
dimensions to 8 × 8. This pattern allows the CNN to trade spatial resolution
for increasingly rich learned features.

The resulting feature maps are passed to the classification head, where they
are flattened into a one-dimensional representation and processed by a dense
neural-network layer. The final softmax layer produces one probability for
each of the ten digit classes.

== Detailed Layer Architecture

While @cnn-architecture presents the model at the component level,
@cnn-architecture-detailed shows the complete layer-by-layer implementation.
The detailed view preserves the same three architectural groups while exposing
the transformations performed inside each block.

The first feature-extraction block contains two convolutional layers with
LeakyReLU activation, followed by max pooling and batch normalization. The
second block repeats the same general pattern while increasing the number of
learned feature maps.

The classification head converts the final #model-final-feature-shape feature-map tensor into
a #model-flattened-features element feature vector. A dense layer reduces this representation to
#model-hidden-dense-units learned features, followed by LeakyReLU activation and dropout
regularization. The final dense layer contains #model-output-classes softmax outputs corresponding
to the digit classes 0 through 9.

#pagebreak()

#figure(
  image(
    "../figures/cnn_architecture_detailed.png",
    height: 80%,
  ),
  caption: [
    Detailed layer-by-layer architecture of the trained SVHN convolutional
    neural network.
  ],
) <cnn-architecture-detailed>

== Model Capacity

The trained CNN contains #model-total-parameters parameters distributed
across #model-number-of-layers layers, resulting in a relatively compact
architecture for an image classification model. Most of the model capacity
is concentrated in the dense classification layer after the convolutional
feature maps are flattened.

The convolutional layers contain substantially fewer parameters because their
filters are shared across spatial locations. In contrast, the first dense layer
connects every element of the flattened feature representation to each of its
output neurons, resulting in the largest parameter contribution within the
network.

This distribution of parameters reflects the different responsibilities of the
two portions of the model: the convolutional blocks extract spatial features,
while the dense layers combine those learned features to perform final
classification.

== Architecture Summary

The table below summarizes the primary structural characteristics of the
trained convolutional neural network.

#figure(
  table(
    columns: (1fr, 1fr),
    inset: 8pt,
    stroke: 0.8pt,

    table.header(
      [*Architecture Property*],
      [*Value*],
    ),

    [Input Shape], [#model-input-shape],
    [Total Layers], [#model-number-of-layers],
    [Total Parameters], [#model-total-parameters],
    [Final Feature Shape], [#model-final-feature-shape],
    [Flattened Features], [#model-flattened-features],
    [Hidden Dense Units], [#model-hidden-dense-units],
    [Output Classes], [#model-output-classes],
    [Convolution Kernel], [#model-kernel-size],
    [Dropout Rate], [#model-dropout-rate],
  ),
  caption: [
    Summary of the trained CNN architecture and model capacity.
  ],
) <architecture-summary>

== Design Decisions

Several architectural choices were used to improve feature learning,
optimization stability, and generalization.

The convolutional layers use #model-kernel-size kernels with same padding, allowing the
network to learn local spatial patterns while preserving feature-map dimensions
until pooling is applied. The number of feature maps increases as the network
becomes deeper, enabling progressively richer visual representations.

LeakyReLU activation is used throughout the hidden layers so that small
negative activations can continue to propagate rather than being forced
entirely to zero. Max-pooling layers reduce spatial dimensionality and
computational cost, while batch normalization helps maintain stable feature
distributions during training.

A dropout layer is included in the classification head to reduce reliance on
individual neurons and improve generalization. Finally, the ten-unit softmax
output layer converts the network output into a probability distribution across
the ten SVHN digit classes.