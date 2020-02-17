# Evaluation Metric of Network

The goal of this evaluation metric is to compare the outputs of a music generation network and output a score of how closely it relates to a dataset of songs composed by humans. Ideally, it should replicate how a human would compare the two sets of songs but that's difficult to guarantee.

## Dataset Used as "Real" Songs

The dataset used can be found here: https://magenta.tensorflow.org/datasets/maestro

## Method of Comparison
A MIDI file is first converted to a vector of numerical features, inspired by this [paper](https://www.researchgate.net/publication/328728367_On_the_evaluation_of_generative_models_in_music?fbclid=IwAR0HTFr7J9em2kIm2BJGblJbiGC3nmCCOp7hQgE_wEems8Px-ZQOnDo6qXI).

| Feature Name | Description |
| --- | --- |
| Pitch Count | Number of different pitches in the piece |
| Pitch Range | The difference between the highest pitch and the lowest pitch in the piece |
| Avg Pitch Interval | Average difference between two consecutive pitches (measured in semitones) |
| # Notes | Number of notes in piece |
| Average Inter-Onset-Interval | The average time between two notes |
| Pitch Class Histogram | Number of times each pitch class was used (i.e. CDEFGBA) |
| Pitch Class Transition Matrix | Number of times each unique pitch class transition occurred (e.g. C -> C, C -> D, G -> A, A -> G) |
| Duration Histogram | Number of notes present with a certain duration (e.g. number of quarter notes) |
| Duration Transition Matrix | Number of times each unique duration transition occurred (e.g. 1/4 note to 1/2 note, whole note to 1/8 note) |

Once these feature vectors are calculated, we evaluate them by training a classifier on “real” songs and the set of songs in question. How well it performs is taken as the “score” of the set of songs. This method allows the evaluator to recognize any type of fake music, since it trains itself on half of the fake music before testing itself. It also allows for more complicated comparison between two songs (beyond Euclidean Distance) but is still relatively simple.

**Results of Exploring Option C**

We need our test to return low scores for all types of randomly generated noise and high scores for songs similar to our dataset.

**Details behind script:**

The real data and the fake data are shuffled and split into a training dataset and a test dataset, where the real songs’ labels are “1” and the fake songs’ labels are “0”. The datasets are then normalized (real and fake together to preserve the difference in distribution) to speed up the training process. The classifier is fit using the sklearn fit() method and then calculates its training accuracy and test accuracy. This process is repeated 5 times and the results are averaged. Generally, the number of fake songs should be about the number of real songs (i.e. ~1282)

Note that using an SVC with a linear kernel is basically trying to separate the two datasets with a best-fit hyperplane which is a nicer description of the process than “using a classifier”.



**SVC(Linear), Test Size 0.5**
<table>
  <tr>
   <td><strong>Real Songs</strong>
   </td>
   <td><strong>Fake Songs</strong>
   </td>
   <td><strong>Training Accuracy</strong>
   </td>
   <td><strong>Test Accuracy</strong>
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset
   </td>
   <td>0.64586583
   </td>
   <td>0.35413417
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,0.1)
   </td>
   <td>0.64274571
   </td>
   <td>0.37597504
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,1)
   </td>
   <td>0.73868955
   </td>
   <td>0.50312012
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0, 10)
   </td>
   <td>0.80811232
   </td>
   <td>0.63572543
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>N(mu, sigma)
   </td>
   <td>0.76521061
   </td>
   <td>0.53822153
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Exp(sigma)
   </td>
   <td>1
   </td>
   <td>0.99765991
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Logistic(mu, sigma)
   </td>
   <td>0.8174727
   </td>
   <td>0.60530421
   </td>
  </tr>
</table>


**SVC(Poly, deg=3, gamma=scale), Test Size 0.5**


<table>
  <tr>
   <td><strong>Real Songs</strong>
   </td>
   <td><strong>Fake Songs</strong>
   </td>
   <td><strong>Training Accuracy</strong>
   </td>
   <td><strong>Test Accuracy</strong>
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset
   </td>
   <td>0.6201248
   </td>
   <td>0.3798752
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,0.1)
   </td>
   <td>0.65054602
   </td>
   <td>0.36895476
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,1)
   </td>
   <td>0.90639626
   </td>
   <td>0.55772231
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0, 10)
   </td>
   <td>1.
   </td>
   <td>0.7550702
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>N(mu, sigma)
   </td>
   <td>1.
   </td>
   <td>0.81669267
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Exp(sigma)
   </td>
   <td>1.
   </td>
   <td>0.97347894
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Logistic(mu, sigma)
   </td>
   <td>1.
   </td>
   <td>0.72776911
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Zeros
   </td>
   <td>0.99063963
   </td>
   <td>0.98205928
   </td>
  </tr>
</table>




**SVC(RBF, gamma=scale), Test Size 0.5**


<table>
  <tr>
   <td><strong>Real Songs</strong>
   </td>
   <td><strong>Fake Songs</strong>
   </td>
   <td><strong>Training Accuracy</strong>
   </td>
   <td><strong>Test Accuracy</strong>
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset
   </td>
   <td>0.6349454
   </td>
   <td>0.3650546
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,0.1)
   </td>
   <td>0.64040562
   </td>
   <td>0.37909516
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,1)
   </td>
   <td>0.94149766
   </td>
   <td>0.89391576
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0, 10)
   </td>
   <td>0.99453978
   </td>
   <td>0.99219969
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>N(mu, sigma)
   </td>
   <td>0.99219969
   </td>
   <td>0.975039
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Exp(sigma)
   </td>
   <td>1
   </td>
   <td>0.99921997
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Logistic(mu, sigma)
   </td>
   <td>1
   </td>
   <td>0.99453978
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Zeros
   </td>
   <td>1
   </td>
   <td>1
   </td>
  </tr>
</table>


**MLPClassifier(), Test Size 0.5**


<table>
  <tr>
   <td><strong>Real Songs</strong>
   </td>
   <td><strong>Fake Songs</strong>
   </td>
   <td><strong>Training Accuracy</strong>
   </td>
   <td><strong>Test Accuracy</strong>
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset
   </td>
   <td>0.74804992
   </td>
   <td>0.25195008
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,0.1)
   </td>
   <td>0.83447738
   </td>
   <td>0.28346334
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,1)
   </td>
   <td>0.99859594
   </td>
   <td>0.71123245
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0, 10)
   </td>
   <td>1
   </td>
   <td>0.84024961
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>N(mu, sigma)
   </td>
   <td>0.99719189
   </td>
   <td>0.85803432
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Exp(sigma)
   </td>
   <td>1
   </td>
   <td>0.99375975
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Logistic(mu, sigma)
   </td>
   <td>0.79859594
   </td>
   <td>0.66053042
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Zeros
   </td>
   <td>1
   </td>
   <td>0.99921997
   </td>
  </tr>
</table>




**LogisticRegression(solver=lbfgs, max_iter=200), Test Size 0.5**


<table>
  <tr>
   <td><strong>Real Songs</strong>
   </td>
   <td><strong>Fake Songs</strong>
   </td>
   <td><strong>Training Accuracy</strong>
   </td>
   <td><strong>Test Accuracy</strong>
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset
   </td>
   <td>0.62948518
   </td>
   <td>0.37051482
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,0.1)
   </td>
   <td>0.63026521
   </td>
   <td>0.38221529
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0,1)
   </td>
   <td>0.70046802
   </td>
   <td>0.47191888
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Dataset + N(0, 10)
   </td>
   <td>0.80577223
   </td>
   <td>0.61856474
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>N(mu, sigma)
   </td>
   <td>0.74336973
   </td>
   <td>0.51482059
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Exp(sigma)
   </td>
   <td>1
   </td>
   <td>0.99765991
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Logistic(mu, sigma)
   </td>
   <td>0.79797192
   </td>
   <td>0.60920437
   </td>
  </tr>
  <tr>
   <td>Dataset
   </td>
   <td>Zeros
   </td>
   <td>1
   </td>
   <td>0.99921997
   </td>
  </tr>
</table>



<!-- Docs to Markdown version 1.0β18 -->
