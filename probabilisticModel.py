from starter4 import *
from sklearn.model_selection import train_test_split
from decimal import Decimal

def probabilisticReasoning(data):

    dropColumns = ["funny, cool, useful"]
    data.drop(columns=["funny", "cool", "useful"], inplace=True)

    train, test = train_test_split(
        data, test_size=0.2, random_state=42)

    #print(df.head())
    star_one_count = len(train[train['stars'] == 1.0])
    star_two_count = len(train[train['stars'] == 2.0])
    star_three_count = len(train[train['stars'] == 3.0])
    star_four_count = len(train[train['stars'] == 4.0])
    star_five_count = len(train[train['stars'] == 5.0])

    # ugly print statements, sorry. These are for debugging the recall accuracy
    print("Test Set, 1 star ratings:", len(test[test['stars'] == 1]))
    print("Test Set, 2 star ratings:", len(test[test['stars'] == 2]))
    print("Test Set, 3 star ratings:", len(test[test['stars'] == 3]))
    print("Test Set, 4 star ratings:", len(test[test['stars'] == 4]))
    print("Test Set, 5 star ratings:", len(test[test['stars'] == 5]))
    print()

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
    # Dictionary containing the probability of a word appearing in a review that has a certain star rating
    # Indexing this at p_word_has_stars["word", 5.0] will give the probability of "word" appearing in a 5.0 review
    p_word_has_stars = JointProbDist(['text', 'stars'])

    # iterate through the rows of pandas DataFrame using the function `iterrows`

    training_progress = 0
    for index, row in train.iterrows():
        #print(row['text'], row['stars'])
        #print(row['text'])
        training_progress += 1
        if training_progress % 50000 == 0:
            print("Training at row:", training_progress)
        try:
            for word in row['text'].split():
                #print(word)
                p_word_has_stars[word, row['stars']] += 1
        except:
            print(row)
            print(row['text'])
            print("---------")

    # Add a smoothing factor of 1 to all counts
    for word in p_word_has_stars.values('text'):
        for category in p_word_has_stars.values('stars'):
            p_word_has_stars[word, category] += 1

    # Add a case for when a word is not in the training set
    p_word_has_stars["not_in_training_set", 1] = 1
    p_word_has_stars["not_in_training_set", 2] = 1
    p_word_has_stars["not_in_training_set", 3] = 1
    p_word_has_stars["not_in_training_set", 4] = 1
    p_word_has_stars["not_in_training_set", 5] = 1

    # OLD NORMALIZATION METHOD, APPARENTLY DOES WORK
    for c in p_word_has_stars.values('stars'):  # for each star rating (category):
        total = sum(                            # total the number of words appearing in that category
            (p_word_has_stars[w, c] for w in p_word_has_stars.values('text'))
        )
        # then divide everything by the total so that for each star rating, the probabilities of it having each word sums to 1
        for w in p_word_has_stars.values('text'):
            p_word_has_stars[w, c] = Decimal(p_word_has_stars[w, c] / total)

    # OTHER NORMALIZATION METHOD, APPARENTLY DOES NOT WORK
    # # normalize the collected counts and turn them to probability distributions over each category
    # # i.e. for each word, the probabilities of it being each star should sum to 1
    # for word in p_word_has_stars.values('text'):
    #     total = p_word_has_stars[word, 1] + p_word_has_stars[word, 2] + p_word_has_stars[word, 3] + p_word_has_stars[word, 4] + p_word_has_stars[word, 5]
    #     p_word_has_stars[word, 1] /= total
    #     p_word_has_stars[word, 2] /= total
    #     p_word_has_stars[word, 3] /= total
    #     p_word_has_stars[word, 4] /= total
    #     p_word_has_stars[word, 5] /= total

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
    # For each of the test instances, calculate the probability of each star rating given the text message
    testing_progress = 0
    for [label, text] in test.values:
        testing_progress += 1
        if testing_progress % 50000 == 0:
            print("Testing at row:", testing_progress)

        p_one = Decimal(stars_prob_dist["one"])
        p_two = Decimal(stars_prob_dist["two"])
        p_three = Decimal(stars_prob_dist["three"])
        p_four = Decimal(stars_prob_dist["four"])
        p_five = Decimal(stars_prob_dist["five"])

        try:
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
                # print(word.ljust(20), "-",
                #       str(p_one.__round__(10)).ljust(12),
                #       str(p_two.__round__(10)).ljust(12),
                #       str(p_three.__round__(10)).ljust(12),
                #       str(p_four.__round__(10)).ljust(12),
                #       str(p_five.__round__(10)).ljust(12))
        except:
            print(label)
            print(text)
            print("---------")

        values = [p_one, p_two, p_three, p_four, p_five]
        #print(values)

        if p_one == max(values):
            if label == 1.0:
                one_predicted_correct += 1
            else:
                one_predicted_incorrect += 1
            total_one_predicted += 1
        elif p_two == max(values):
            if label == 2.0:
                two_predicted_correct += 1
            else:
                two_predicted_incorrect += 1
            total_two_predicted += 1
        elif p_three == max(values):
            if label == 3.0:
                three_predicted_correct += 1
            else:
                three_predicted_incorrect += 1
            total_three_predicted += 1
        elif p_four == max(values):
            if label == 4.0:
                four_predicted_correct += 1
            else:
                four_predicted_incorrect += 1
            total_four_predicted += 1
        elif p_five == max(values):
            if label == 5.0:
                five_predicted_correct += 1
            else:
                five_predicted_incorrect += 1
            total_five_predicted += 1


    # once your prediction is ready for each instance, increment the proper equivalent values from the 6 values above
    # (for each instace only one *predicted_as* variable will be updated and one *total_* variable depending on
    # the actual test message label.

    print(f"Predicted One: {total_one_predicted}\n"
          f"Predicted Two: {total_two_predicted}\n"
          f"Predicted Three: {total_three_predicted}\n"
          f"Predicted Four: {total_four_predicted}\n"
          f"Predicted Five: {total_five_predicted}\n")

    # print("confusion matrix\tprd_one\tprd_two\tprd_three\tprd_four\tprd_five\t"
    #       "nact_one\tnact_two\tnact_three\tnact_four\tnact_five\t\t{}\t\t{}\n".format(
    #     one_predicted_correct, two_predicted_correct, three_predicted_correct,
    #     four_predicted_correct, five_predicted_correct, one_predicted_incorrect,
    #     two_predicted_incorrect, three_predicted_incorrect, four_predicted_incorrect, five_predicted_incorrect
    # ))

    acc_one = (one_predicted_correct * 100 / total_one_predicted) if total_one_predicted != 0 else 0
    acc_two = (two_predicted_correct * 100 / total_two_predicted) if total_two_predicted != 0 else 0
    acc_three = (three_predicted_correct * 100 / total_three_predicted) if total_three_predicted != 0 else 0
    acc_four = (four_predicted_correct * 100 / total_four_predicted) if total_four_predicted != 0 else 0
    acc_five = (five_predicted_correct * 100 / total_five_predicted) if total_five_predicted != 0 else 0

    rec_one = (one_predicted_correct * 100 / len(test[test['stars'] == 1])) if (
            len(test[test['stars'] == 1])) != 0 else 0
    rec_two = (two_predicted_correct * 100 / len(test[test['stars'] == 2])) if (
            len(test[test['stars'] == 2])) != 0 else 0
    rec_three = (three_predicted_correct * 100 / len(test[test['stars'] == 3])) if (
            len(test[test['stars'] == 3])) != 0 else 0
    rec_four = (four_predicted_correct * 100 / len(test[test['stars'] == 4])) if (
            len(test[test['stars'] == 4])) != 0 else 0
    rec_five = (five_predicted_correct * 100 / len(test[test['stars'] == 5])) if (
            len(test[test['stars'] == 5])) != 0 else 0

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
    print(f"Average F1:\t\t\t{(f1_one+f1_two+f1_three+f1_four+f1_five)/5}")