# Glossary

Every important term in the handbook, defined briefly. Chapters define each term properly the first time it appears. Use this page as a quick reminder.

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [Z](#z)

## A

A/B test
:   A randomized experiment that compares two versions (A, the control, and B, the treatment) by randomly assigning units such as users to each, then measuring the difference in a metric.

Accuracy
:   The fraction of predictions that are correct. Misleading on imbalanced data: predicting "not fraud" every time can be 99.9% accurate.

Activation function
:   A nonlinear function applied to a neuron's weighted input, such as ReLU, sigmoid, tanh, or GELU. Without nonlinearity, a stack of layers collapses into one linear map.

Actor-critic
:   An RL method that learns both a policy (the actor) and a value function (the critic), using the critic to reduce the variance of policy-gradient updates.

Adam
:   An optimizer that keeps running averages of the gradient (first moment) and the squared gradient (second moment) to give each parameter an adaptive step size. **AdamW** decouples weight decay from the gradient update.

Attention
:   A mechanism that computes a weighted average of value vectors, with weights given by how well a query matches each key: $\mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$.

AUC (area under the curve)
:   Usually the area under the ROC curve: the probability that a randomly chosen positive example is scored higher than a randomly chosen negative one.

Autoencoder
:   A network trained to reconstruct its input through a narrow bottleneck, which forces it to learn a compressed representation.

Autograd
:   Automatic differentiation: software that records the operations of a computation and applies the chain rule backward to compute exact gradients.

## B

Backpropagation
:   The algorithm that computes the gradient of the loss with respect to every parameter by applying the chain rule backward through the computational graph, reusing intermediate results.

Bagging (bootstrap aggregating)
:   Training many models on bootstrap samples of the data and averaging their predictions, which reduces variance.

Batch normalization
:   Normalizing each feature of a layer's activations using the mean and variance of the current mini-batch, then applying a learned scale and shift.

Batch size
:   The number of examples used to compute one gradient estimate and parameter update.

Bayes' theorem
:   $P(A \mid B) = P(B \mid A)\,P(A) / P(B)$. It updates a prior belief $P(A)$ with evidence $B$ to get a posterior.

Bias (statistical)
:   The difference between an estimator's expected value and the true value. In the bias-variance trade-off, the error from a model being too simple to capture the true pattern.

Bias (in a model)
:   The intercept term $b$ in $\mathbf{w}^\top\mathbf{x} + b$, which lets the output shift independently of the input.

Bias-variance trade-off
:   Expected prediction error splits into bias², variance, and irreducible noise. Simpler models tend to have high bias and low variance; flexible models have the reverse.

Bootstrap
:   Resampling the data with replacement many times to estimate the sampling distribution of a statistic, for example to build a confidence interval.

BPE (byte-pair encoding)
:   A subword tokenization method that starts from characters or bytes and repeatedly merges the most frequent adjacent pair into a new token.

Broadcasting
:   NumPy's and PyTorch's rules for combining arrays of different shapes by virtually stretching dimensions of size 1, without copying data.

## C

Calibration
:   How well predicted probabilities match observed frequencies. Among examples scored 0.8, about 80% should be positive.

Categorical variable
:   A feature that takes values from a fixed set of categories, such as country or plan type.

Causal inference
:   Methods for estimating the effect of an intervention, rather than mere association, from experimental or observational data.

Central limit theorem
:   The sum or mean of many independent random variables with finite variance is approximately normally distributed, whatever their original distribution.

Chain rule
:   The derivative of a composition: $\frac{d}{dx} f(g(x)) = f'(g(x))\,g'(x)$. The mathematical core of backpropagation.

Class imbalance
:   When one class is much rarer than the others, as in fraud or disease detection.

Classification
:   Supervised learning where the target is a category.

CLIP
:   A model trained contrastively to place matching images and text captions close together in a shared embedding space.

Clustering
:   Grouping unlabeled examples so that examples in the same group are similar, for example with k-means or DBSCAN.

Computational graph
:   A directed graph whose nodes are operations and whose edges carry values. Autograd records it on the forward pass and traverses it backward.

Concept drift
:   A change over time in the relationship between the inputs and the target, so a trained model's learned mapping goes stale.

Confidence interval
:   A range computed from data by a procedure that, over repeated samples, contains the true parameter a stated fraction of the time, such as 95%.

Confounder
:   A variable that influences both the treatment and the outcome, creating a spurious association between them.

Confusion matrix
:   A table of counts of true positives, false positives, true negatives, and false negatives.

Contrastive learning
:   Self-supervised learning that pulls representations of related pairs (two views of the same image, an image and its caption) together and pushes unrelated pairs apart.

Convex function
:   A function where any line segment between two points on its graph lies on or above the graph. Any local minimum is a global minimum.

Convolution
:   Sliding a small learned kernel over an input and computing a dot product at each position. It shares weights across positions and detects local patterns.

Correlation
:   A measure of association between two variables. Pearson correlation measures linear association; Spearman correlation measures monotonic association.

Cross-entropy
:   $H(p, q) = -\sum_x p(x) \log q(x)$. As a loss, the negative log-probability the model assigns to the true class.

Cross-validation
:   Estimating generalization by splitting the data into $k$ folds, training on $k-1$ folds, evaluating on the remaining one, and averaging over all $k$ rounds.

Curse of dimensionality
:   Phenomena in high-dimensional spaces, such as distances becoming similar and data becoming sparse, that make many methods (especially distance-based ones) work poorly.

## D

Data augmentation
:   Creating modified copies of training examples (crops, flips, noise) to increase effective dataset size and improve robustness.

Data leakage
:   When information that won't be available at prediction time, such as the target or future data, leaks into training. It produces impressive validation scores that collapse in production.

DataFrame
:   pandas' two-dimensional labeled table: columns of possibly different types sharing a row index.

DBSCAN
:   A density-based clustering algorithm that grows clusters from dense regions and labels isolated points as noise.

Decision boundary
:   The surface in feature space where a classifier's prediction changes from one class to another.

Decision tree
:   A model that predicts by asking a sequence of feature-threshold questions, learned by recursively splitting the data to reduce impurity.

Deep learning
:   Machine learning with neural networks of many layers that learn hierarchical representations directly from raw data.

Diffusion model
:   A generative model trained to reverse a gradual noising process. It generates data by starting from noise and denoising step by step.

Dimensionality reduction
:   Mapping data to fewer dimensions while preserving important structure, for example with PCA, t-SNE, or UMAP.

Distillation
:   Training a small "student" model to match the outputs of a large "teacher" model.

DPO (direct preference optimization)
:   A method for aligning language models to human preferences directly from preference pairs, without training a separate reward model or running RL.

Dropout
:   A regularizer that randomly zeroes a fraction of activations during training, which discourages co-adaptation.

## E

Early stopping
:   Halting training when validation performance stops improving. A simple and effective regularizer.

Eigenvector
:   A nonzero vector $\mathbf{v}$ that a matrix only scales: $A\mathbf{v} = \lambda \mathbf{v}$, where the scalar $\lambda$ is its eigenvalue.

ELBO (evidence lower bound)
:   A tractable lower bound on the log-likelihood that variational autoencoders maximize. It combines a reconstruction term and a KL-divergence term.

Embedding
:   A learned dense vector representing a discrete item (a word, a user, a product) such that similar items have nearby vectors.

EM (expectation-maximization)
:   An algorithm for models with hidden variables that alternates between estimating the hidden variables (E-step) and re-fitting the parameters (M-step). It's used to fit Gaussian mixture models.

Ensemble
:   A model that combines the predictions of several models, as in bagging, boosting, or stacking.

Entropy
:   $H(X) = -\sum_x p(x) \log p(x)$: the average uncertainty, or information content, of a random variable.

Epoch
:   One full pass over the training dataset.

Expected value
:   The probability-weighted average of a random variable's values.

Exploratory data analysis (EDA)
:   Systematically summarizing and visualizing data to understand it, find problems, and form hypotheses before modeling.

## F

F1 score
:   The harmonic mean of precision and recall: $2PR/(P+R)$.

False positive / false negative
:   A false positive is a negative predicted as positive (a Type I error). A false negative is a positive predicted as negative (a Type II error).

Feature
:   An input variable to a model, also called a predictor or attribute.

Feature engineering
:   Creating, transforming, and selecting input features to make patterns easier for a model to learn.

Fine-tuning
:   Continuing to train a pretrained model on a new, usually smaller, task-specific dataset.

FSDP (fully sharded data parallel)
:   A distributed training method that shards parameters, gradients, and optimizer state across GPUs to train models too large for one GPU.

## G

GAN (generative adversarial network)
:   A generator and a discriminator trained against each other: the generator tries to produce realistic samples, and the discriminator tries to tell real from fake.

Gaussian mixture model (GMM)
:   A model of data as coming from a weighted mix of several Gaussian distributions, usually fit with EM.

Generalization
:   How well a model performs on new, unseen data from the same distribution as its training data.

Gini impurity
:   $1 - \sum_k p_k^2$: the probability of misclassifying a randomly drawn example if it were labeled randomly according to the node's class distribution. Used to choose decision tree splits.

Gradient
:   The vector of partial derivatives of a function. It points in the direction of steepest increase.

Gradient boosting
:   Building an ensemble sequentially, where each new model (usually a small tree) fits the negative gradient of the loss of the current ensemble.

Gradient clipping
:   Scaling down gradients whose norm exceeds a threshold, to prevent exploding updates.

Gradient descent
:   Iteratively minimizing a function by stepping opposite the gradient: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$.

## H

Hallucination
:   When a language model generates fluent but false or unsupported content.

Hinge loss
:   $\max(0, 1 - y\,f(\mathbf{x}))$ for labels $y \in \{-1, +1\}$: the loss behind support vector machines.

Hyperparameter
:   A setting chosen before training rather than learned from data, such as learning rate, tree depth, or regularization strength.

Hypothesis test
:   A procedure for deciding whether data provide enough evidence against a null hypothesis, usually using a p-value.

## I

Imputation
:   Filling in missing values, for example with the median, a model's prediction, or a constant plus a "was missing" indicator.

In-context learning
:   A language model performing a task from instructions or examples given in its prompt, without any weight updates.

Inference
:   In ML: using a trained model to make predictions. In statistics: drawing conclusions about a population from a sample.

Information gain
:   The reduction in entropy achieved by a split. Decision trees choose splits that maximize it.

Isolation forest
:   An anomaly detection method that isolates points with random splits. Anomalies are isolated in fewer splits.

## K

k-fold cross-validation
:   See *cross-validation*.

k-means
:   Clustering that alternates between assigning each point to its nearest centroid and moving each centroid to the mean of its points.

k-nearest neighbors (kNN)
:   Predicting from the labels of the $k$ closest training examples.

Kernel trick
:   Computing inner products in a high-dimensional feature space implicitly, through a kernel function, without ever constructing the features.

KL divergence
:   $D_{\mathrm{KL}}(p \parallel q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$: how much one distribution differs from another. It's not symmetric.

KV cache
:   During autoregressive generation, storing the keys and values of past tokens so they aren't recomputed at each step.

## L

L1 / L2 regularization
:   Adding a penalty to the loss: the sum of absolute weights (L1, lasso), which drives some weights to exactly zero, or the sum of squared weights (L2, ridge), which shrinks all weights.

Label
:   The target value a supervised model learns to predict.

Language model
:   A model that assigns probabilities to sequences of tokens, usually by predicting the next token given the previous ones.

Layer normalization
:   Normalizing activations across the features of each individual example. The standard choice in transformers.

Learning curve
:   A plot of training and validation performance against training set size (or training time), used to diagnose bias and variance.

Learning rate
:   The step size $\eta$ in gradient descent. The single most important hyperparameter in deep learning.

Likelihood
:   The probability (or density) of the observed data as a function of the model parameters.

Linear regression
:   Predicting a continuous target as a weighted sum of features plus a bias, usually fit by least squares.

Logistic regression
:   A linear classifier that passes a weighted sum through a sigmoid to output a probability, trained by minimizing log-loss.

Logit
:   The raw, unnormalized score before a sigmoid or softmax. For a probability $p$, $\mathrm{logit}(p) = \log\frac{p}{1-p}$.

LoRA (low-rank adaptation)
:   Parameter-efficient fine-tuning that freezes the pretrained weights and learns small low-rank update matrices $BA$ added to them.

Loss function
:   A function measuring how wrong a model's predictions are. Training minimizes it.

LSTM (long short-term memory)
:   A recurrent unit with gates that control what to remember, forget, and output, which mitigates vanishing gradients over long sequences.

## M

Machine learning
:   Building systems that improve at a task by learning patterns from data rather than following hand-written rules.

MAE / MSE / RMSE
:   Mean absolute error, mean squared error, and root mean squared error: regression metrics that measure average prediction error.

Matrix factorization
:   Approximating a matrix as a product of smaller matrices. Used in recommender systems to learn user and item embeddings.

Maximum likelihood estimation (MLE)
:   Choosing the parameters that make the observed data most probable.

Mini-batch
:   A small random subset of the training data used for one gradient update.

Mixed precision
:   Training with lower-precision numbers (fp16 or bf16) for most operations, while keeping critical values in fp32, to save memory and time.

MLOps
:   The practices and tools for deploying, monitoring, and maintaining ML models in production reliably.

Momentum
:   An optimizer technique that accumulates a velocity from past gradients to smooth updates and speed progress along consistent directions.

Multi-head attention
:   Running several attention operations in parallel on different learned projections and concatenating the results, so each head can attend to different relationships.

## N

Naive Bayes
:   A probabilistic classifier that applies Bayes' theorem with the "naive" assumption that features are conditionally independent given the class.

Neural network
:   A model built from layers of units, each computing a weighted sum followed by a nonlinearity, trained with gradient descent.

Normal distribution
:   The bell-shaped Gaussian distribution, defined by its mean $\mu$ and variance $\sigma^2$.

Normalization (of features)
:   Rescaling features, for example to zero mean and unit variance (standardization) or to the range [0, 1] (min-max scaling).

Null hypothesis
:   The default assumption a hypothesis test tries to reject, usually "no effect" or "no difference".

## O

One-hot encoding
:   Representing a category as a vector with a 1 in that category's position and 0 everywhere else.

Optimizer
:   The algorithm that updates parameters from gradients, such as SGD, momentum, RMSProp, or Adam.

Outlier
:   An observation far from the bulk of the data. It may be an error, or the most interesting point in the dataset.

Overfitting
:   When a model learns noise and quirks of the training data, so it performs well on training data but poorly on new data.

## P

p-value
:   The probability, assuming the null hypothesis is true, of observing data at least as extreme as what was observed. It is *not* the probability that the null hypothesis is true.

Parameter
:   A value the model learns from data, such as a weight or a bias.

Perceptron
:   The original single-neuron linear classifier, trained with a simple mistake-driven update rule.

Perplexity
:   The exponential of the average per-token cross-entropy of a language model. Lower is better.

Pipeline
:   A chain of preprocessing steps and a model, fit together so that preprocessing is learned only from training data, which prevents leakage.

Policy gradient
:   RL methods that directly adjust a policy's parameters in the direction that increases expected reward, as in REINFORCE and PPO.

Positional encoding
:   Information about token positions added to transformer inputs, since attention by itself ignores order.

PPO (proximal policy optimization)
:   A policy-gradient algorithm that limits how far each update can move the policy, for stable training. Used in RLHF.

Precision
:   Of the examples predicted positive, the fraction that really are positive: $TP/(TP+FP)$.

Principal component analysis (PCA)
:   Projecting data onto the orthogonal directions of maximum variance, computed from the covariance matrix's eigenvectors or the data's SVD.

Prior / posterior
:   In Bayesian inference, the belief before seeing data (the prior) and the updated belief after (the posterior).

## Q

Q-learning
:   An RL algorithm that learns the value $Q(s, a)$ of taking action $a$ in state $s$, using the Bellman equation as an update target.

Quantization
:   Storing and computing with lower-precision numbers, such as 8-bit or 4-bit integers instead of 32-bit floats, to shrink models and speed up inference.

## R

RAG (retrieval-augmented generation)
:   Retrieving relevant documents, usually by embedding similarity, and adding them to a language model's prompt so its answer is grounded in them.

Random forest
:   An ensemble of decision trees, each trained on a bootstrap sample with a random subset of features considered at each split, whose predictions are averaged.

Recall
:   Of the actual positives, the fraction the model found: $TP/(TP+FN)$. Also called sensitivity or the true positive rate.

Regression
:   Supervised learning where the target is a continuous number.

Regularization
:   Any technique that constrains a model to reduce overfitting, such as weight penalties, dropout, early stopping, or data augmentation.

Reinforcement learning (RL)
:   Learning to act by trial and error, where an agent takes actions in an environment to maximize cumulative reward.

ReLU
:   The rectified linear unit, $\max(0, x)$: the default activation function in most deep networks.

Residual connection
:   Adding a layer's input to its output ($x + f(x)$), which eases gradient flow and makes very deep networks trainable.

RLHF (reinforcement learning from human feedback)
:   Fine-tuning a language model with RL against a reward model trained on human preference comparisons.

RNN (recurrent neural network)
:   A network that processes a sequence one step at a time, carrying a hidden state forward.

ROC curve
:   A plot of true positive rate against false positive rate as the decision threshold varies.

## S

Sampling (from a language model)
:   Choosing the next token from the model's predicted distribution, controlled by temperature, top-k, or top-p (nucleus) settings.

Scaling laws
:   Empirical power-law relationships between model performance and model size, dataset size, and compute.

Self-attention
:   Attention in which the queries, keys, and values all come from the same sequence, so every position can attend to every other.

Self-supervised learning
:   Learning representations from unlabeled data by creating labels from the data itself, such as predicting masked words or the next token.

SHAP
:   A method that attributes a prediction to the features using Shapley values from cooperative game theory.

Sigmoid
:   $\sigma(z) = 1/(1 + e^{-z})$: squashes any real number into (0, 1).

Simpson's paradox
:   A trend that appears in several groups but reverses or disappears when the groups are combined.

Softmax
:   $\mathrm{softmax}(\mathbf{z})_i = e^{z_i} / \sum_j e^{z_j}$: turns a vector of scores into a probability distribution.

Stationarity
:   A time series property: the statistical properties (mean, variance, autocorrelation) don't change over time.

Stochastic gradient descent (SGD)
:   Gradient descent using the gradient from a single example or mini-batch rather than the whole dataset.

Stratified sampling
:   Splitting data so each split has the same class proportions as the whole dataset.

Supervised learning
:   Learning a mapping from inputs to known target labels.

Support vector machine (SVM)
:   A classifier that finds the boundary with the maximum margin between classes, optionally in a kernel-induced feature space.

SVD (singular value decomposition)
:   Factoring any matrix as $U \Sigma V^\top$: a rotation, a scaling along orthogonal axes, and another rotation.

## T

t-SNE
:   A nonlinear technique for visualizing high-dimensional data in 2D or 3D that preserves local neighborhoods. Distances between clusters in its plots aren't meaningful.

Teacher forcing
:   Training a sequence model by feeding it the true previous token, rather than its own prediction, at each step.

Tensor
:   A multidimensional array: the core data structure in PyTorch.

Test set
:   Data held out until the very end, used only once to estimate final performance.

TF-IDF
:   Term frequency × inverse document frequency: a word's weight in a document, high when it's frequent in that document but rare across documents.

Token
:   The unit of text a language model processes: a word, a subword, a character, or a byte.

Transfer learning
:   Reusing a model trained on one task as the starting point for another.

Transformer
:   A neural network architecture built from self-attention and feed-forward layers with residual connections and normalization. The basis of modern LLMs.

## U

UMAP
:   A nonlinear dimensionality reduction method, often faster than t-SNE, that tends to preserve more global structure.

Underfitting
:   When a model is too simple to capture the underlying pattern, so it performs poorly even on training data.

Universal approximation theorem
:   A neural network with one hidden layer and enough units can approximate any continuous function on a bounded domain to any accuracy. It says nothing about how to find the weights.

Unsupervised learning
:   Finding structure in data without labels, as in clustering, dimensionality reduction, and density estimation.

## V

Validation set
:   Data held out from training and used to choose hyperparameters and models.

Vanishing / exploding gradients
:   Gradients that shrink toward zero or grow without bound as they're multiplied backward through many layers, which stalls or destabilizes training.

Variance (statistical)
:   The expected squared deviation from the mean. In the bias-variance trade-off, how much a model's predictions change across different training sets.

VAE (variational autoencoder)
:   A probabilistic autoencoder that learns a smooth latent space you can sample from, trained by maximizing the ELBO.

Vectorization
:   Replacing Python loops with whole-array operations that run in optimized compiled code.

Vision transformer (ViT)
:   A transformer applied to images by splitting them into patches and treating each patch as a token.

## W

Weight decay
:   Shrinking weights toward zero at each step. It's equivalent to L2 regularization for plain SGD, but not for Adam, which is why AdamW exists.

Window function (SQL)
:   A SQL function that computes a value over a set of rows related to the current row, such as a running total or a rank, without collapsing the rows.

word2vec
:   A method that learns word embeddings by predicting words from their contexts (CBOW) or contexts from words (skip-gram).

## Z

Zero-shot learning
:   Performing a task without any task-specific training examples, for example by following instructions in a prompt or by matching against text descriptions as in CLIP.
