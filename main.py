###############################################################
### Martin Van Laethem  - Aiden Lonquist    - Cole Atkinson ###
### A01189452           - A01166561         - A00000000     ###
###            AI Project 2 - Data Prediction               ###
###############################################################

import pandas as pd
import json
from sklearn.model_selection import train_test_split
import numpy as np
from starter4 import *

def DoTheYoinkySploinky():
    # Read the data
    data = []
    dataTypes = {
        #"review_id": str,
        #"user_id": str,
        #"business_id": str,
        #"date": str,
        "stars": np.float16,
        "useful": np.int32,
        "funny": np.int32,
        "cool": np.int32,
    }
    with open("yelp_academic_dataset_review.json", "r", encoding="utf-8") as file:
        reader = pd.read_json(file, orient="records", lines=True, dtype=dataTypes, chunksize=100000)

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

    print(data.head())

    return data, dataTypes

def probModel(data, dataTypes):

    data.drop(columns=["funny", "cool", "useful"], inplace=True)

    train, test = train_test_split(
        data, test_size=0.2, random_state=42)

    #print(df.head())
    star_zero_count = len(train[train['stars'] == 0.0])
    star_one_count = len(train[train['stars'] == 1.0])
    star_two_count = len(train[train['stars'] == 2.0])
    star_three_count = len(train[train['stars'] == 3.0])
    star_four_count = len(train[train['stars'] == 4.0])
    star_five_count = len(train[train['stars'] == 5.0])
    print(star_zero_count, star_one_count, star_two_count, star_three_count, star_four_count, star_five_count)
    # # TODO continue from here and create p_category here using the train data
    # # use show_approx function and make sure the probability of spam is not too low here (e.g. below 0.11)
    # # if it was the case re-run the 'train_test_split' cell!
    p_cat = {"zero": star_zero_count, "one": star_one_count, "two": star_two_count, "three": star_three_count, "four": star_four_count, "five": star_five_count}
    stars_prob_dist = ProbDist("stars", p_cat)
    print(stars_prob_dist.__getitem__("zero"))
    print(stars_prob_dist.__getitem__("one"))
    print(stars_prob_dist.__getitem__("two"))
    print(stars_prob_dist.__getitem__("three"))
    print(stars_prob_dist.__getitem__("four"))
    print(stars_prob_dist.__getitem__("five"))


    # TODO create p_has_word_category as a JointProbDist and fill it in by iterating through train instances
    p_has_word_category = JointProbDist(['text', 'stars'])
    # iterate through the rows of pandas DataFrame using the function `iterrows`
    for index, row in train.iterrows():
        #print(row['text'], row['stars'])
        # print(row['TEXT'].split())
        for word in row['text'].split():
            #print(word)
            p_has_word_category[word, row['stars']] += 1

    #print(p_has_word_category.values("stars"))
    # normalize the collected counts and turn them to probability distributions over each category
    # i.e. the content of p_has_word_category for each category must sum to 1.
    #print(p_has_word_category["Worst", 3])

    for c in p_has_word_category.values('stars'):
        total = sum((p_has_word_category[w, c] for w in p_has_word_category.values('text')))
        for w in p_has_word_category.values('text'):
            p_has_word_category[w, c] /= total

    #print(p_has_word_category["Worst", 3])

    # for c in p_has_word_category.values('stars'):
    #     print("Sum probability for category {} should sum to 1. The actual summation is equal to {}.".format(
    #         c, sum((p_has_word_category[w, c] for w in p_has_word_category.values('text')))))

    zero_predicted_correct = 1
    zero_predicted_incorrect = 1
    total_zero_predicted = 1
    one_predicted_correct = 1
    one_predicted_incorrect = 1
    total_one_predicted = 1
    two_predicted_correct = 1
    two_predicted_incorrect = 1
    total_two_predicted = 1
    three_predicted_correct = 1
    three_predicted_incorrect = 1
    total_three_predicted = 1
    four_predicted_correct = 1
    four_predicted_incorrect = 1
    total_four_predicted = 1
    five_predicted_correct = 1
    five_predicted_incorrect = 1
    total_five_predicted = 1
    # #############################################################################################
    # IMPORTANT! DO NOT MODIFY THE LINES ABOVE THIS LINE

    # TODO use the Naïve Bayes classification equation here to classify test data.

    # Implementation hint: simply use the two distributions you just collected and calculate P(ham|text_message)
    # and P(spam|text_message) and selected the one with higher probability as the message class.
    #print(test.values)

    for [label, text] in test.values:
        p_zero = stars_prob_dist['zero']
        p_one = stars_prob_dist['one']
        p_two = stars_prob_dist['two']
        p_three = stars_prob_dist['three']
        p_four = stars_prob_dist['four']
        p_five = stars_prob_dist['five']
        for word in text.split():
            p_zero *= p_has_word_category[word, 'zero']
            p_one *= p_has_word_category[word, 'one']
            p_two *= p_has_word_category[word, 'two']
            p_three *= p_has_word_category[word, 'three']
            p_four *= p_has_word_category[word, 'four']
            p_five *= p_has_word_category[word, 'five']

        values = [p_zero, p_one, p_two, p_three, p_four, p_five]

        if p_zero == max(values):
            if label == 'zero':
                zero_predicted_correct += 1
            else:
                zero_predicted_incorrect += 1
            total_zero_predicted += 1
        elif p_one == max(values):
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

    # Once you are done with the implementation, running this cell will use your collected stats and print out the
    # confusion matrix and precision, recall, and f-1 scores of your classifier.

    # IMPORTANT! DO NOT MODIFY THE LINES BELOW THIS LINE
    # #############################################################################################
    print("confusion matrix\tprd_zero\tprd_one\tprd_two\tprd_three\tprd_four\tprd_five\tnact_zero\t"
          "nact_one\tnact_two\tnact_three\tnact_four\tnact_five\t\t{}\t\t{}\n".format(
        zero_predicted_correct, one_predicted_correct, two_predicted_correct, three_predicted_correct,
        four_predicted_correct, five_predicted_correct, zero_predicted_incorrect, one_predicted_incorrect,
        two_predicted_incorrect, three_predicted_incorrect, four_predicted_incorrect, five_predicted_incorrect
    ))

    acc_zero = zero_predicted_correct * 100 / total_zero_predicted
    acc_one = one_predicted_correct * 100 / total_one_predicted
    acc_two = two_predicted_correct * 100 / total_two_predicted
    acc_three = five_predicted_correct * 100 / total_three_predicted
    acc_four = five_predicted_correct * 100 / total_four_predicted
    acc_five = five_predicted_correct * 100 / total_five_predicted
    rec_zero = zero_predicted_correct * 100 / (zero_predicted_correct + zero_predicted_incorrect)
    rec_one = one_predicted_correct * 100 / (one_predicted_correct + one_predicted_incorrect)
    rec_two = two_predicted_correct * 100 / (two_predicted_correct + two_predicted_incorrect)
    rec_three = three_predicted_correct * 100 / (three_predicted_correct + three_predicted_incorrect)
    rec_four = four_predicted_correct * 100 / (four_predicted_correct + four_predicted_incorrect)
    rec_five = five_predicted_correct * 100 / (five_predicted_correct + five_predicted_incorrect)
    f1_zero = 2 * acc_zero * rec_zero / (acc_zero + rec_zero)
    f1_one = 2 * acc_one * rec_one / (acc_one + rec_one)
    f1_two = 2 * acc_two * rec_two / (acc_two + rec_two)
    f1_three = 2 * acc_three * rec_three / (acc_three + rec_three)
    f1_four = 2 * acc_four * rec_four / (acc_four + rec_four)
    f1_five = 2 * acc_five * rec_five / (acc_five + rec_five)
    print("Prediction accuracy\tzero = {:.3f}\tone = {:.3f}\ttwo = {:.3f}\tthree = {:.3f}\tfour = {:.3f}\tfive = {:.3f}"
          .format(acc_zero, acc_one, acc_two, acc_three, acc_four, acc_five))
    print("Prediction recall\tzero = {:.3f}\tone = {:.3f}\ttwo = {:.3f}\tthree = {:.3f}\tfour = {:.3f}\tfive = {:.3f}"
          .format(rec_zero, rec_one, rec_two, rec_three, rec_four, rec_five))
    print("Prediction F1\t\tzero = {:.3f}\tone = {:.3f}\ttwo = {:.3f}\tthree = {:.3f}\tfour = {:.3f}\tfive = {:.3f}"
          .format(f1_zero, f1_one, f1_two, f1_three, f1_four, f1_five))

if __name__ == "__main__":
    data, dataTypes = DoTheYoinkySploinky()
    probModel(data, dataTypes)
