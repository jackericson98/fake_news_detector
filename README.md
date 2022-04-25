# Fake News Detector


## How to run:
1. Install the required packages: pandas, sklearn using pip:
```
$ pip install pandas, sklearn
```
2. Open the command prompt and move to the working directory:
```
$ cd /path/to/directory/
```
4.  Run the program and simulation:
```
$ python detector.py
```
5. Create detector object with the news data file:
```
>>> myDetector = NewsClassifier('news.csv')
```
6. Train the model and test its predictions accuracy:
```
>>> myDetector.train()
>>> myDetector.predict()
```

## Data format:
The data needs to be in a comma seperated file (.csv) file format, in the following order:\
number, title, text, label

## Methods
This program uses a large dataset of news articles with Fake and Real designations to train a model to be able to predict the validity of other news articles. It starts by filtering out very common words and then giving frequency weights to each term in each article. If the term is common in the article it gets a higher score for that article, but, on the flip side, if the word is universally common its score goes down. With the keywords extracted, the algorithm is ready to be trained.

The detector also relies on a passive aggressive classifier algorithm, as the driver for its training. This type of algorithm does not react when the classification is correct, but reacts heavily if the classification is wrong. This trains the model to be really good at making sure no fake news is classified as real, but has the trade off of occasionally classifying some real news articles as fake.


