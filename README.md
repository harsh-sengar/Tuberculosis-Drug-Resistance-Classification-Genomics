# Tuberculosis-Drug-Resistance-Classification-Genomics [Kaggle](https://www.kaggle.com/competitions/tuberculosis-drug-resistance-prediction-rif/overview#)

# Description
The objective of the assignment is to produce a machine learning/statistical model that improves the predictive performance of Whole Genome Sequencing (WGS) for a set of first and second line drugs. Provided a set of mutations as features, we are to fit a model on the data provided and predict whether a sample is resistant or susceptible to a particular drug. Eleven drugs were taken into consideration which are first-line drugs (rifampicin, isoniazid, pyrazinamide and ethambutol); streptomycin; second-line injectable drugs (capre- omycin, amikacin, and kanamycin); and fluoroquinolones (ciprofloxacin, moxifloxacin, and ofloxacin). This report provides a description of the steps taken in creating the models, the processing of datasets and the results inferred from the trained models. The report also provides a reasoning and analysis of the models performance over the datasets.<br/>
# Required libraries
* Numpy
* Matplotlib
* Pandas
* Tensorflow
* Sklearn
* XGBoost
<br/>
Make sure to install the above libraries and finally run the scripts

# Datasets utilized
* **Row deletion**<br/><br/>
  The method of imputation is as the title describes it to be. Delete all samples which have missing feature values in the training data. Although the imputation method reduces the number of samples available for training, the method allows us to capture true, experimentally generated information provided by every feature in the dataset.
* **Column deletion**<br/><br/>
  On observing the dataset, it was found that all the missing values fell within three particular features, (SNP CN 2714366 C967A V323L eis, SNP I 2713795 C329T inter Rv2415c eis and SNP I 2713872 C252A inter Rv2415c eis) due to which the three features were removed as the presence of large number of missing values seemed to outweigh the information provided by the three features.<br/><br/>
* **Simple oversampling**<br/><br/>
Copies of the samples of the minority class were made to prevent the class from being neglected while the model is trained. The point of choosing the simple oversampling technique was to check if a simpler solution would work better than a more complex solution.<br/><br/>
* **Synthetic Minority Oversampling Technique (SMOTE)**<br/><br/>
SMOTE selects samples that are close in the feature space, draws a line between the samples in the feature space and draws a new sample at a point along that line, i.e. a random sample from the minority class is first chosen. Then k of the nearest neighbors for that sample are found. A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two samples in the feature space. The method works as it causes the classifier to build greater decision regions that contain nearby minority class points. A con of this approach is that the samples are created without considering the majority class which could result in contradictory samples if there is a strong overlap for the classes. Smote was applied using the python library imblearn.<br/><br/>
* **logistic PCA**<br/><br/>
An extension over PCA logistic PCA is utilized to extend PCA to a binary dataset. What logistic regression is to linear regression, logistic PCA is to PCA.
This is done so by projecting the natural parameters from the Bernoulli saturated model and minimizing the Bernoulli deviance. In this feature set, a value of K = 155 was chosen. This decision was based over the visualization provided by the scree plot. 155 features accounted for ∼97% of the total variance found in the dataset. The dataset was created using the R library logistic PCA where the parameters of K = 155 and m = 14 was considered (the value of m is chosen for which the negative log likelihood is the least after applying the method cv.lpca over multiple values of m).<br/><br/>
* **Chi-square**<br/><br/>
Among several uses of the chi-square test, we used the chi-square test for feature selection. We defined our Null hypothesis as "the feature is independent (or
does not provide sufficient information) for the output variable". Chi-square value of each of the input features is calculated and using the p-value, the calculated chi-square value is checked (whether it falls in the accepted or rejected region with the help of the chi-square table). Accordingly features were selected or rejected to obtain the final dataset.<br/><br/>
# Models Utilized
* **Logistic regression:**<br/><br/> 
Logistic regression is a statistical model which is used primarily for classification tasks. Logistic regression uses an equation similar to the equation of linear regression, however, it models the dependent (or output) variable as a binary (or categorical) value using a logistic or sigmoid function [4]. A L2 regularized logistic regression model has been used for the experiments.<br/><br/>
* **Support Vector Machine:**<br/><br/>
Support Vector Machines (or SVM) is a supervised machine learning algorithm which can be used for both classification and regression tasks. For our experiments, we will be using Support Vector Classification (or SVC). The intuitive idea behind SVM is to find the optimal decision boundary (or hyper-plane) that separates the two classes. A Gaussian kernel implementation using the radial basis function has been used.<br/><br/>
* **Neural network:**<br/><br/>
Inside the human brain, multiple neurons are interconnected, where each neuron takes a sensory input and produces a response. Similarly in an artificial neural network, each input node is connected to numerous neurons in multiple hidden layers, which are in turn are connected to the output nodes in the output layers. The layers are fully connected, and each interconnection has a weight associated with it, which are learned during the training process. Back propagation and gradient descent algorithm are used to train the neural net.<br/><br/>
* **Extreme gradient boosting (XGBoost):**<br/><br/>
It is an ensemble learning technique based on gradient boosted decision trees, where errors made by the older models are corrected by adding new models. In gradient boosting the models are generated to predict the errors of the previous models, and they are added sequentially to make the final prediction. This process of adding models continues until no further improvements are possible. In extreme gradient boosting, gradient descent algorithm is used to optimize (minimize) the loss function.<br/><br/>
* **Super Learner Ensemble (SLE):**<br/><br/>
The super learner algorithm is based off of the stacking ensemble technique. It involves pre-defining a k-fold split of the training dataset, then evaluating all algorithms (models in the ensemble) and algorithm configurations on the same split of the data. All out-of-fold predictions are then used to train a meta-model that learns how to best combine the predictions. We stacked SVM and XGBoost as base models and used Logistic regression as the meta-model.<br/><br/>
**Note: For information on the observations and results gained through the assignment refer the report provided**
