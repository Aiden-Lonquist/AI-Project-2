import random
import string
import time
import math
from tqdm import tqdm

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALL_LETTERS = string.ascii_letters + "1234567890 .,;'-$é\n"
N_LETTERS = len(ALL_LETTERS) + 1  # Plus EOS marker
ALL_RATINGS = [1.0, 2.0, 3.0, 4.0, 5.0]
criterion = nn.NLLLoss()

learning_rate = 0.0005
hidden_layer_size = 128
n_iters = 1000  # default 100000
print_every = 5000
plot_every = 500


def RNNMain(data):

    learning_rate = 0.0005
    hidden_layer_size = 128
    n_iters = 100_000
    print_every = 5000
    plot_every = 500


    # rnn = RecurrentNeuralNetwork(N_LETTERS, hidden_layer_size, N_LETTERS)

    all_losses = []
    total_loss = 0  # Reset every plot_every iters

    start = time.time()

    i_tqdm = tqdm(range(1, n_iters + 1))
    for iter in i_tqdm:
        rating_tensor, current_letter_tensor, next_letter_tensor = getRandomTrainingExample(data)
        next_letter_tensor.unsqueeze_(-1)
        hidden = rnn.initHidden()

        rnn.zero_grad()

        loss = 0

        for i in range(current_letter_tensor.size(0)):
            output, hidden = rnn(rating_tensor, current_letter_tensor[i], hidden)
            try:
                l = criterion(output, next_letter_tensor[i])
            except:
                print("Command executed:", next_letter_tensor[i])
                print("i value that caused exception:", i, "Current Letter Tensor size:", next_letter_tensor.size(0))

            loss += l

        loss.backward()

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        output, loss = output, loss.item() / current_letter_tensor.size(0)
        total_loss += loss

        i_tqdm.set_description('Loss: %.4f' % (loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    plt.figure()
    plt.plot(all_losses)

    samples(1, 'b')

    samples(2, 'def')

    samples(3, 'ghi')

    samples(4, 'jkl')

    samples(5, 'mno')


# Random item from a list or dataframe
def randomChoice(l):
    if type(l) != list:
        return l.iloc[random.randint(0, len(l) - 1)]
    else:
        return l[random.randint(0, len(l) - 1)]


# Get a random rating value and random review with that rating
def randomTrainingPair(data):
    r_rating = randomChoice(ALL_RATINGS)
    r_review = randomChoice(data.loc[data['stars'] == r_rating])["text"]
    return r_rating, r_review


# This helper function will take a `time.time()` and returns the time difference since that time
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def currentLetterTensor(review):
    if type(review) != str:
        review = str(review)
    result = torch.zeros(len(review), 1, N_LETTERS).float()
    for letter_position in range(len(review)):
        result[letter_position][0][ALL_LETTERS.find(review[letter_position])] = 1
    return result.to(device)


def ratingTensor(rating):
    result = torch.zeros(1, len(ALL_RATINGS)).float()
    result[0][ALL_RATINGS.index(rating)] = 1
    return result.to(device)


def nextLetterTensor(review):
    result = [ALL_LETTERS.find(review[letter_position]) for letter_position in range(1, len(review))]
    # for letter_position in range(1, len(review)):
    #     result = [ALL_LETTERS.find(review[letter_position])]
    #     if ALL_LETTERS.find(review[letter_position]) == -1:
    #         print("THIS CHARACTER BREAKS IT", review[letter_position], "<- THAT ONE RIGHT THERE. position:", letter_position, "review:", review)
    #         print("Space index:", ALL_LETTERS.find(' '))
    result.append(N_LETTERS - 1) # EOS
    for i in range(0, len(result)):
        if result[i] == -1:
            result[i] = 62
    return torch.LongTensor(result).to(device)


def getRandomTrainingExample(data):
    r_rating, r_review = randomTrainingPair(data)
    language_tensor = ratingTensor(r_rating)
    current_letter_tensor = currentLetterTensor(r_review)
    next_letter_tensor = nextLetterTensor(r_review)
    return language_tensor, current_letter_tensor, next_letter_tensor


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecurrentNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        n_categories = len(ALL_RATINGS)
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, language, input, hidden):
        input_combined = torch.cat((language, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)


def sample(rating, start_letter='A', max_length = 200):
    with torch.no_grad():  # no need to track history in sampling
        rating_tensor = ratingTensor(rating)
        input = currentLetterTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(rating_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == N_LETTERS - 1:
                break
            else:
                letter = ALL_LETTERS[topi]
                output_name += letter
            input = currentLetterTensor(letter)

        return output_name


def samples(rating, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(rating, start_letter))


rnn = RecurrentNeuralNetwork(N_LETTERS, hidden_layer_size, N_LETTERS)