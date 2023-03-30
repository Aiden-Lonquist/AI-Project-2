###############################################################
### Martin Van Laethem  - Aiden Lonquist    - Cole Atkinson ###
### A01189452           - A01166561         - A00000000     ###
###            AI Project 2 - Data Prediction               ###
###############################################################

import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import numpy as np
from starter4 import *
from normalize import *

def DoTheYoinkySploinky() -> pd.DataFrame:
    # Read the data
    data = []
    dataTypes = {
        "stars": np.float16,
        "useful": np.int32,
        "funny": np.int32,
        "cool": np.int32,
    }
    with open("testData.json", "r", encoding="utf-8") as file:
        reader = pd.read_json(file, orient="records", lines=True, dtype=dataTypes, chunksize=32)

        # Filter/clean the data
        for chunk in reader:
            #np.where(pd.isnull(chunk))
            # missing_cols, missing_rows = (
            #     (chunk.isnull().sum(x) | chunk.eq('').sum(x))
            #     .loc[lambda x: x.gt(0)].index
            #     for x in (0, 1)
            # )

            #print(chunk.loc[missing_rows, missing_cols])
            reducedChunk = chunk.drop(columns=["review_id", "user_id", "business_id", "date"])

            data.append(reducedChunk)

    # Concatenate the data to put all chunks together in one dataframe
    data = pd.concat(data)

    #data = pd.read_csv("testData.csv", dtype=dataTypes)

    #print(data.head())

    return data

def probModel(data):

    data.drop(columns=["funny", "cool", "useful"], inplace=True)

    train, test = train_test_split(
        data, test_size=0.2, random_state=42)

    #print(df.head())
    star_one_count = len(train[train['stars'] == 1.0])
    star_two_count = len(train[train['stars'] == 2.0])
    star_three_count = len(train[train['stars'] == 3.0])
    star_four_count = len(train[train['stars'] == 4.0])
    star_five_count = len(train[train['stars'] == 5.0])
    print("Total counts and probabilities:")
    print(f"1*: {star_one_count},\t 2*: {star_two_count},\t 3*: {star_three_count},\t 4*: {star_four_count},\t 5*: {star_five_count}")

    # Create a ProbDist object for the stars category
    p_cat = {"one": star_one_count, "two": star_two_count, "three": star_three_count, "four": star_four_count, "five": star_five_count}
    stars_prob_dist = ProbDist("stars", p_cat)
    print(f"1*: {stars_prob_dist['one'].__round__(3)},"
          f"\t 2*: {stars_prob_dist['two'].__round__(3)},"
          f"\t 3*: {stars_prob_dist['three'].__round__(3)},"
          f"\t 4*: {stars_prob_dist['four'].__round__(3)},"
          f"\t 5*: {stars_prob_dist['five'].__round__(3)}")


    # Create p_word_has_stars as a JointProbDist and fill it in by iterating through train instances
    # Dictionary containing the probability of a review having a certain star rating given that it contains a certain word
    p_word_has_stars = JointProbDist(['text', 'stars'])

    #print(p_word_has_stars['hi'])

    # iterate through the rows of pandas DataFrame using the function `iterrows`
    for index, row in train.iterrows():
        #print(row['text'], row['stars'])
        # print(row['TEXT'].split())
        for word in row['text'].split():
            #print(word)
            p_word_has_stars[word, row['stars']] += 1

    # # Add a smoothing factor of 1 to all counts
    # for c in p_word_has_stars.values('stars'):
    #     for w in p_word_has_stars.values('text'):
    #         p_word_has_stars[w, c] += 1

    # Add a case for when a word is not in the training set
    p_word_has_stars["not_in_training_set", 1] = 1
    p_word_has_stars["not_in_training_set", 2] = 1
    p_word_has_stars["not_in_training_set", 3] = 1
    p_word_has_stars["not_in_training_set", 4] = 1
    p_word_has_stars["not_in_training_set", 5] = 1

    # # OLD NORMALIZATION FUNCTION, DOES NOT WORK I THINK
    # for c in p_word_has_stars.values('stars'):
    #     total = sum((p_word_has_stars[w, c] for w in p_word_has_stars.values('text')))
    #     for w in p_word_has_stars.values('text'):
    #         p_word_has_stars[w, c] /= total

    # normalize the collected counts and turn them to probability distributions over each category
    # i.e. for each word, the probabilities of the stars should sum to 1
    for key in p_word_has_stars.values('text'):
        total = p_word_has_stars[key, 1] + p_word_has_stars[key, 2] + p_word_has_stars[key, 3] + p_word_has_stars[key, 4] + p_word_has_stars[key, 5]
        p_word_has_stars[key, 1] /= total
        p_word_has_stars[key, 2] /= total
        p_word_has_stars[key, 3] /= total
        p_word_has_stars[key, 4] /= total
        p_word_has_stars[key, 5] /= total
        #print(p_word_has_stars[key, 1], p_word_has_stars[key, 2], p_word_has_stars[key, 3], p_word_has_stars[key, 4], p_word_has_stars[key, 5])

    one_predicted_correct = 0
    one_predicted_incorrect = 0
    total_one_predicted = 0
    two_predicted_correct = 0
    two_predicted_incorrect = 0
    total_two_predicted = 0
    three_predicted_correct = 0
    three_predicted_incorrect = 0
    total_three_predicted = 0
    four_predicted_correct = 0
    four_predicted_incorrect = 0
    total_four_predicted = 0
    five_predicted_correct = 0
    five_predicted_incorrect = 0
    total_five_predicted = 0


    # Use the Naïve Bayes classification equation here to classify test data.

    # Implementation hint: simply use the two distributions you just collected and calculate P(ham|text_message)
    # and P(spam|text_message) and selected the one with higher probability as the message class.

    # For each of the test instances, calculate the probability of each star rating given the text message
    for [label, text] in test.values:
        p_one = stars_prob_dist["one"]
        p_two = stars_prob_dist["two"]
        p_three = stars_prob_dist["three"]
        p_four = stars_prob_dist["four"]
        p_five = stars_prob_dist["five"]
        for word in text.split():
            if word not in p_word_has_stars.values('text'):
                word = "not_in_training_set"
            # else:
            #     print(p_word_has_stars[word, 1], p_word_has_stars[word, 2], p_word_has_stars[word, 3], p_word_has_stars[word, 4], p_word_has_stars[word, 5])
            p_one *= p_word_has_stars[word, 1]
            p_two *= p_word_has_stars[word, 2]
            p_three *= p_word_has_stars[word, 3]
            p_four *= p_word_has_stars[word, 4]
            p_five *= p_word_has_stars[word, 5]

            print(word.ljust(20), "-",
                  str(p_one.__round__(10)).ljust(12),
                  str(p_two.__round__(10)).ljust(12),
                  str(p_three.__round__(10)).ljust(12),
                  str(p_four.__round__(10)).ljust(12),
                  str(p_five.__round__(10)).ljust(12))

        values = [p_one, p_two, p_three, p_four, p_five]
        #print(values)

        if p_one == max(values):
            if label == 'one':
                one_predicted_correct += 1
            else:
                one_predicted_incorrect += 1
            total_one_predicted += 1
        elif p_two == max(values):
            if label == 'two':
                two_predicted_correct += 1
            else:
                two_predicted_incorrect += 1
            total_two_predicted += 1
        elif p_three == max(values):
            if label == 'three':
                three_predicted_correct += 1
            else:
                three_predicted_incorrect += 1
            total_three_predicted += 1
        elif p_four == max(values):
            if label == 'four':
                four_predicted_correct += 1
            else:
                four_predicted_incorrect += 1
            total_four_predicted += 1
        elif p_five == max(values):
            if label == 'five':
                five_predicted_correct += 1
            else:
                five_predicted_incorrect += 1
            total_five_predicted += 1


    # once your prediction is ready for each instance, increment the proper equivalent values from the 6 values above
    # (for each instace only one *predicted_as* variable will be updated and one *total_* variable depending on
    # the actual test message label.

    print(f"One: {total_one_predicted}\nTwo: {total_two_predicted}\nThree: {total_three_predicted}\nFour: {total_four_predicted}\nFive: {total_five_predicted}\n")

    print("confusion matrix\tprd_one\tprd_two\tprd_three\tprd_four\tprd_five\t"
          "nact_one\tnact_two\tnact_three\tnact_four\tnact_five\t\t{}\t\t{}\n".format(
        one_predicted_correct, two_predicted_correct, three_predicted_correct,
        four_predicted_correct, five_predicted_correct, one_predicted_incorrect,
        two_predicted_incorrect, three_predicted_incorrect, four_predicted_incorrect, five_predicted_incorrect
    ))

    acc_one = (one_predicted_correct * 100 / total_one_predicted) if total_one_predicted != 0 else 0
    acc_two = (two_predicted_correct * 100 / total_two_predicted) if total_two_predicted != 0 else 0
    acc_three = (five_predicted_correct * 100 / total_three_predicted) if total_three_predicted != 0 else 0
    acc_four = (five_predicted_correct * 100 / total_four_predicted) if total_four_predicted != 0 else 0
    acc_five = (five_predicted_correct * 100 / total_five_predicted) if total_five_predicted != 0 else 0
    rec_one = (one_predicted_correct * 100 / (one_predicted_correct + one_predicted_incorrect)) if (
            one_predicted_correct + one_predicted_incorrect) != 0 else 0
    rec_two = (two_predicted_correct * 100 / (two_predicted_correct + two_predicted_incorrect)) if (
            two_predicted_correct + two_predicted_incorrect) != 0 else 0
    rec_three = (three_predicted_correct * 100 / (three_predicted_correct + three_predicted_incorrect)) if (
            three_predicted_correct + three_predicted_incorrect) != 0 else 0
    rec_four = (four_predicted_correct * 100 / (four_predicted_correct + four_predicted_incorrect)) if (
            four_predicted_correct + four_predicted_incorrect) != 0 else 0
    rec_five = (five_predicted_correct * 100 / (five_predicted_correct + five_predicted_incorrect)) if (
            five_predicted_correct + five_predicted_incorrect) != 0 else 0
    f1_one = (2 * acc_one * rec_one / (acc_one + rec_one)) if (acc_one + rec_one) != 0 else 0
    f1_two = (2 * acc_two * rec_two / (acc_two + rec_two)) if (acc_two + rec_two) != 0 else 0
    f1_three = (2 * acc_three * rec_three / (acc_three + rec_three)) if (acc_three + rec_three) != 0 else 0
    f1_four = (2 * acc_four * rec_four / (acc_four + rec_four)) if (acc_four + rec_four) != 0 else 0
    f1_five = (2 * acc_five * rec_five / (acc_five + rec_five)) if (acc_five + rec_five) != 0 else 0
    print("Prediction accuracy\tone = {:.3f}\ttwo = {:.3f}\tthree = {:.3f}\tfour = {:.3f}\tfive = {:.3f}"
          .format(acc_one, acc_two, acc_three, acc_four, acc_five))
    print("Prediction recall\tone = {:.3f}\ttwo = {:.3f}\tthree = {:.3f}\tfour = {:.3f}\tfive = {:.3f}"
          .format(rec_one, rec_two, rec_three, rec_four, rec_five))
    print("Prediction F1\t\tone = {:.3f}\ttwo = {:.3f}\tthree = {:.3f}\tfour = {:.3f}\tfive = {:.3f}"
          .format(f1_one, f1_two, f1_three, f1_four, f1_five))

def normalizeDataframe(data: pd.DataFrame) -> pd.DataFrame:
    # Pass each text field to the normalize function
    for index, row in data.iterrows():
        data.at[index, 'text'] = normalize(row['text'])

    return data


if __name__ == "__main__":


    if os.path.exists('normalized_data.csv'):
        data = pd.read_csv('normalized_data.csv')
    else:
        data = DoTheYoinkySploinky()
        data = normalizeDataframe(data)
        # save the normalized data to a file
        data.to_csv('normalized_data.csv', index=False)

    probModel(data)
