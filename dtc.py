from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.naive_bayes import MultinomialNB



def NaiveBayes(data, target:str):
    data_one = (data[data['stars'] == 1]).head(10000)
    data_two = (data[data['stars'] == 2]).head(10000)
    data_three = (data[data['stars'] == 3]).head(10000)
    data_four = (data[data['stars'] == 4]).head(10000)
    data_five = (data[data['stars'] == 5]).head(10000)

    data = pd.concat([data_one, data_two, data_three, data_four, data_five])

    tokenized_text = []

    for index, line in data.iterrows():
        try:
            tokenized_text.append(simple_preprocess(line['text'], deacc=True))
        except Exception as e:
            tokenized_text.append(simple_preprocess(" "))

    data['tokenized_text'] = tokenized_text

    # print(data)

    bow_transformer = CountVectorizer().fit(data['text'])

    # print(len(bow_transformer.vocabulary_))
    #
    # print(bow_transformer.transform(data.loc(1)['text']))

    data_reviews = bow_transformer.transform(data['text'])

    # print('Shape of Sparse Matrix: ', data_reviews.shape)
    # print('Amount of Non-Zero occurrences: ', data_reviews.nnz)
    # # Percentage of non-zero values
    # density = (100.0 * data_reviews.nnz / (data_reviews.shape[0] * data_reviews.shape[1]))
    # print('Density: {}'.format((density)))

    trainX, testX, trainY, testY = train_test_split(data_reviews, data['stars'], test_size=0.2, random_state=42)


    print(trainX.count())
    print(trainY.count())


    nb = MultinomialNB()
    nb.fit(trainX, trainY)

    preds = nb.predict(testX)

    print("Classification report:\n", classification_report(testY, preds, zero_division=0))

def DTC(data, target:str):
    data_one = (data[data['stars'] == 1]).head(50000)
    data_two = (data[data['stars'] == 2]).head(50000)
    data_three = (data[data['stars'] == 3]).head(50000)
    data_four = (data[data['stars'] == 4]).head(50000)
    data_five = (data[data['stars'] == 5]).head(50000)

    data = pd.concat([data_one, data_two, data_three, data_four, data_five])

    # Split the data into training and testing data
    dropColumns = ["funny", "cool", "useful", "stars"]
    x = data.drop(columns=dropColumns)  # features
    y = data[target]                    # target
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=42)


    # Preprocess the data
    le = preprocessing.LabelEncoder()
    trainX = trainX.apply(le.fit_transform)
    testX = testX.apply(le.fit_transform)

    # Create the classifier
    clf = tree.DecisionTreeClassifier()

    # Train the classifier
    clf.fit(trainX, trainY)

    # Predict the target
    predicted_target = clf.predict(testX)


    # Print the confusion matrix
    cm = confusion_matrix(testY, predicted_target)
    # cm = cm.diagonal() / cm.sum(axis=1) # get accuracy for each class
    print("Confusion matrix:\n", cm)

    # Print the classification report
    print("Classification report:\n", classification_report(testY, predicted_target, zero_division=0))

    # Print the accuracy
    print("Accuracy:", accuracy_score(testY, predicted_target))
    # Print the recall
    print("Recall:", recall_score(testY, predicted_target, average='macro', zero_division=0))
    # Print the F1 score
    print("F1 score:", f1_score(testY, predicted_target, average='macro', zero_division=0))
