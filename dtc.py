from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
from sklearn import preprocessing



def DTC(data, target:str):

    # Split the data into training and testing data
    dropColumns = ["funny", "cool", "useful", "stars"]
    dropColumns.remove(target)
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
