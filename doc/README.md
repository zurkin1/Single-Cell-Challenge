# Dream Challenge
## Machine Learning in BioInformatics
####

This notebook handles the 60, 40 and 20 genes sub-challenges. It uses a combination of two models.

#### Requirements

- [Python 3.6](https://www.python.org/download/releases/3.6/) --> 3.6
- [numpy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](http://keras.io)

### High Level Solution Overview

**First Model - Max(MCC)**

- First model is based on calculating MAX(MCC) using only 60(40 or 20) genes as opposed to using 84 genes.
- Calculating of MCC is done using matrix multiplication.
- A list of 'candidates' for locations is assembeled using the MAX(MCC) calculation.
- This list is then refined using the second model.
- In the case of 60 genes, MAX(MCC) gives a very good results (location prediction). The second model is hardly needed in this case.

**Second Model - ANN**

- The second model is an ANN to forecast BTDNP sequences given a DGE sequence.
- It automatically learns 'biological' relationships between the two sequences, not necessarily related to MCC accuracy.
- Input: a row from binarized DGE.
- Output: a prediction for a row from binarized BDTNP (the correct location).
- It is used to 'correct' the MAX(MCC) results.
- The advatage of this model is being able to predict correct gene patterns (as opposed to just maximizing MCC, i.e. location).
- The model is relyies on a correct selction of subsets of 60/40/20 genes.
- In the case of 20 genes - it is the only model used since the MAX(MCC) is totally off.

**Combining the Models**

- How to combine the two models?
- We have 10 possibilities (locations) for each cell. We let Max(MCC) propose candidates and then 'correct' the result and select the best candidates using the ANN model.
- If Max(MCC) propose less than 10 results - it means these are very strong results, and we keep them. Otherwise we ignore the results and use only ANN model.
- A manual calibration was done to decide how many candidates we want from the Max(MCC) model. This means selecting the 'cutoff' value of MCC such that we take all locations above this value as a candidate for a location.
- In case of 60 genes trial and error gives an optimal selection of the 2'nd MCC score as a cutoff.
- In case of 40 genese optimal solution is taking the top 2'nd score using Max(MCC) as a cutoff.
- In case of 20 genes all 10 locations are decided using ANN (we are not using the Max(MCC) model at all.)

**Running The Code**
- Make sure you installed Python 3 with SKLearn (we used Anaconda), Tensorflow and Keras.
- Run the cells in the notebook one by one.
- The notebook has to be run three times - for the 60, 40 and 20 genes sub-challenge.
- Manual configuration is only one:  configure num_situ as the number of in-situ genes (sub-challenge) to use. Either 60, 40 or 20. This is done in the first cell.
- The output of the cell before the last will be a submission CSV file.
