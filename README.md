# Fake News Detector
This program uses a large dataset of news articles with Fake and Real designations to train a model to be able to predict the validity of other news articles. The detector trains on the majority of the data (train_size) and tests itself on the remaining articles. 

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
myDetector = NewsClassifier('news.csv')
```
6. Train the model and test its predictions accuracy:
```
myDetector.train()
myDetector.predict()
```
