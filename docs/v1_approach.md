# V1 Approach (End-to-End, in Words)

This is a plain-language description of the V1 approach from start to finish, focusing on what happens conceptually rather than referencing any code or files.

## 1. Task framing
The task is few-shot sound event detection for animal vocalizations. For each recording, you are given five example calls (shots) of a single target class. The system must detect all remaining occurrences of that class in the same recording.

## 2. Data and supervision
The development data has two parts:
- Training data has full multi-class annotations (POS/NEG/UNK) across many species and call types.
- Validation data has a single class per file, with only POS/UNK labels, and the first five POS events are the support shots.

The evaluation data is similar to validation: single class per file, with only the first five POS events provided.

## 3. Features and representation
Audio is converted into time-frequency features, typically log-mel spectrograms (or optionally PCEN). These feature arrays are what the model actually consumes. Each training “example” is a short segment of these features extracted around an event.

## 4. Episodic training (the core idea)
Training is done in episodes. Each episode is a small classification task designed to mimic few-shot conditions.

An episode proceeds as follows:
1. Randomly select N classes (here N = 10).
2. For each class, sample K support segments and K query segments (here K = 5).
3. Embed every segment using a neural network.
4. Create a prototype for each class by averaging the support embeddings.
5. Classify each query embedding by distance to the prototypes and compute the loss.

This encourages the model to build an embedding space where examples from the same class cluster tightly around a prototype, and different classes are well separated.

## 5. Negative contrast training
If negative contrast is enabled, each positive example is paired with a negative segment sampled from the non-event gaps in the same file. This effectively creates an additional “negative” label for each class.

So in each episode:
- You still select 10 classes.
- You still sample 10 items per class (5 support, 5 query).
- But each item now produces two segments: a positive and a negative.

This doubles the number of embedded segments per episode and teaches the model to discriminate true calls from background or non-call regions. Prototypes are built for both positive and negative labels, and the model learns to push positives closer to the correct positive prototype and away from negatives.

## 6. Model architecture (V1)
The V1 model is a convolutional encoder inspired by a ResNet-style structure. It maps each feature segment into a fixed-dimensional embedding (here 2048 dimensions). The classifier is non-parametric: it uses distances to class prototypes rather than a learned softmax layer.

## 7. Distance-based classification
During training, each query embedding is compared to all class prototypes using a distance metric (Euclidean by default). A softmax over negative distances yields class probabilities. The loss encourages the correct class to have the smallest distance.

## 8. Training loop summary
Over many episodes, the model learns an embedding space where:
- Support examples for a class average into a strong prototype.
- Query examples from the same class fall near that prototype.
- Other classes (and negatives) fall farther away.

## 9. Validation and evaluation behavior
Validation uses the same few-shot assumption as evaluation: the first five positive events per file are used as support, and the rest are predicted. Each file is treated independently because the class is unknown and may overlap with other files.

## 10. Overall takeaway
V1 is a prototypical network trained episodically. It learns a universal embedding for bioacoustic calls so that, given only five examples of an unseen class, it can form a prototype and detect similar events in the rest of the recording. Negative contrast strengthens the model by explicitly modeling “not-the-class” regions within the same recordings.
