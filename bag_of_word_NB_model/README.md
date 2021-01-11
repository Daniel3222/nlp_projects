# Text classification of scientific abstract papers with Python

This zipped folder which contains this README file is the result of the Kaggle Data Competetion that just ended in IFT6390. It contains the following :

 - README
 - naive_bayes.py
 - train.csv
 - test.csv
 - english_stopwords.txt

The python file is the script that will output the classification labels for the test data. It uses the train data (train.csv) as input for training and the stopwords file (english_stopwords.txt) as part of the training.

# Bernoulli Naive Bayes classifier : naive_b_classif.py
This python file is structured this way : The creation of the NB Bernoulli classifier, then some useful pre-processing functions and method for transorming text data into vectors (bag-of-words). Then the code that generates the predictions and the outputed csv file.

## NB Bernoulli Class
The Naive Bayes Bernoulli implementation came from this tutorial [Bernoulli NB tutorial].
## Functions

#### The BOW method
 - The BOW that we use in this python script comes from this source : [bag-of-words method] 
 - It creates a sparse matrix of 0 and counts.
 
#### Process
 -  The next part is the process function that came from the lab, it cleans the data of some noise.


## Training and Testing

- The next parts of the script  `<naive_b_classif.py>` loads the data, cleans it , makes the feature enginneering, generates the actual vectors and we train a classifier with these . Lastly, we make the predictions from the training data with the testing data after having proceeded to cleaning, de-noising, feature selection and creating the bag-of-words.
- The file english_stopwords.txt needs to stay in the same folder as all the other files.

## Creating CSV file

- Lastly, we put the data in a pandas dataframe that we will use as a data structure to create our csv file.

### Installation
Install the dependencies and devDependencies and start the server.

```sh
$ pip install numpy
$ pip install pandas
```

Now in order to create the submission.csv file that will contain the labels predicted, there are some important steps.
First, from your terminal, go into the folder containing all scripts and files. Then, one can simply write the following command

```sh
$ python path_to_submission_folder\submission_folder\naive_b_classif.py
```

The output of this line in the terminal will be the creation of a file called `<submission_ift_6390_danielgp.csv>`


**Enjoy!**

   [bag-of-words method]: <https://maelfabien.github.io/machinelearning/NLP_2/#1-preprocessing-per-document-within-corpus>
   [Bernoulli NB tutorial]: <https://kenzotakahashi.github.io/naive-bayes-from-scratch-in-python.html>
   
