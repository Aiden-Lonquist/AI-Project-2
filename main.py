###############################################################
### Martin Van Laethem  - Aiden Lonquist    - Cole Atkinson ###
### A01189452           - A01166561         - A00000000     ###
###            AI Project 2 - Data Prediction               ###
###############################################################

import pandas as pd
import os
from normalize import *
from probabilisticModel import *
from ffnn import *
from dtc import *
import time


def DoTheYoinkySploinky(useFile: str="yelp_academic_dataset_review.json") -> pd.DataFrame:
    # Read the data
    data = []
    dataTypes = {
        "stars": np.float16,
        "useful": np.int32,
        "funny": np.int32,
        "cool": np.int32,
    }

    with open(useFile, "r", encoding="utf-8") as file:
        reader = pd.read_json(file, orient="records", lines=True, dtype=dataTypes, chunksize=1_000_000)

        # Filter/clean the data
        for chunk in reader:
            #np.where(pd.isnull(chunk))
            # missing_cols, missing_rows = (
            #     (chunk.isnull().sum(x) | chunk.eq('').sum(x))
            #     .loc[lambda x: x.gt(0)].index
            #     for x in (0, 1)
            # )

            #print(chunk.loc[missing_rows, missing_cols])

            # Remove the columns that are not considered in the prediction
            reducedChunk = chunk.drop(columns=["review_id", "user_id", "business_id", "date"])

            data.append(reducedChunk)

    # Concatenate the data to put all chunks together in one dataframe
    data = pd.concat(data)

    #data = pd.read_csv("testData.csv", dtype=dataTypes)

    #print(data.head())

    return data

def probModel(data, target: str="stars"):
    print("Starting Probabilistic to predict", target)

    if target == "stars":
        probModelStars(data)
    else:
        #TODO: check if target is a valid column
        probModelEndorsements(data, target)

def FFNNModel(data):
    print("starting FFNN")

    FFNNMain(data)

def DTCModel(data, target: str="stars"):
    print("Starting DTC to predict ", target)

    NaiveBayes(data, target)


def normalizeDataframe(data: pd.DataFrame) -> pd.DataFrame:
    # Pass each text field to the normalize function
    for index, row in data.iterrows():
        data.at[index, 'text'] = normalize(row['text'])

    print("done normalizing")
    return data


if __name__ == "__main__":
    startTime = time.time()

    if os.path.exists('normalized_data.csv'):
        data = pd.read_csv('normalized_data.csv')
    else:
        data = DoTheYoinkySploinky()
        data = normalizeDataframe(data)
        # save the normalized data to a file
        data.to_csv('normalized_data.csv', index=False)

    # data = DoTheYoinkySploinky(useFile="testData.json")

    # probModel(data, target="useful")
    # FFNNModel(data)
    DTCModel(data, target="stars")

    print("\nFinished in",time.strftime("%H:%M:%S", time.gmtime(time.time() - startTime)))
