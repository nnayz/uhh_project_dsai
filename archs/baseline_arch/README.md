### Prototypical Networks

Method of few shot classification, a scenario where a classifier must generalise to new classes not seen during training, given only a small number of examples for each new class. The core concept is to learn a metric space using a neural network, where classfication can be performed by computing distances to a single prototype representation for each class. 

Process is:
- A neural network acts as an embedding function, mapping input data into a vector space. 
- For each class, protype is computed by taking the mean of the embedded support (example) vectors belonging to that class. 
- A new query point is classified by embedding it into the same space and finding the nearest class prototype, typically using a distance function like a squared Euclidean distance. A softmax function is applied over the distances to produce a probability distribution over the classes. 

More about prototype: 

A single vector representation that serves as a central point or centroid for a particular class within a learned embedding space. The fundamental idea is that all examples of a given class should cluster around this representative point. This approach provided a simple and efficient way to represent class, regardless of how many support examples are available. 

Computing prototype requires two main steps:
- Embedding the Support Set: First, small set of labeled examples for a class, known as a "support set" is passed through an embedding function, which is typically a neural network. This function maps each high dimensional input example (e.g., image) into a lower dimensional vector in the metric space. 
- Calculating the Mean Vector: The prototype for class _k_, denoted as $c_k$
, is then calculated as the element-wise mean (or average) of all the embedded support vectors belonging to that class.

$c_k = \frac{1}{\vert S_k\vert } \sum_{(x_i, y_i) \in S_k} f_{\phi}(x_i)$

where $\vert S_k\vert $ is the number of support examples in class $k$, and $f_{\phi}(x_i)$ is the embedded vector of a support example $x_i$

**Choice of using mean**

- Not arbitrary
- As the distance function used for classification is a Bregman divergence, such as the squared Euclidean distance, the mean of a set of points is the optimal representative that minimises the total distance to all points in that set.  [Basically ]