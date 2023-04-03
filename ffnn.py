import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
import pandas as pd
from sys import exit

# https://medium.com/swlh/sentiment-classification-using-feed-forward-neural-network-in-pytorch-655811a0913f

#device = torch.device('cpu')
# if you are using a gpu or you want your code be flexibly running over both CPU and GPU use the following line instead:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def FFNNMain(data):
    # Tokenize the text column to get the new column 'tokenized_text'
    # try:
    #     data['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in data['text']]
    # except Exception as e:
    #     print(e)

    # for line in data['text']:
    #     data['tokenized_text'][line] = simple_preprocess(line, deacc=True)
        #data.loc[line]['tokenized_text'] = simple_preprocess(line, deacc=True)

    tokenized_text = []

    for index, line in data.iterrows():
        try:
            tokenized_text.append(simple_preprocess(line['text'], deacc=True))
        except Exception as e:
            tokenized_text.append(simple_preprocess(" "))


    data['tokenized_text'] = tokenized_text

    print(data)

    print(data['tokenized_text'].head(5))

    # porter_stemmer = PorterStemmer()
    # # Get the stemmed_tokens
    # data['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in data['tokenized_text']]
    # data['stemmed_tokens'].head(10)
    #data['stemmed_tokens'] = data['tokenized_text']

    train_X, test_X, train_Y, test_Y = train_test_split(data[["text", "tokenized_text"]], data["stars"], test_size=0.2, random_state=42)
    # print("train X", train_X.head(5))
    # print("test X", test_X.head(5))
    #
    # print("train Y", train_Y.head(5))
    # print("test Y", test_Y.head(5))

    print(type(train_X))

    review_dict = make_dict(data)

    VOCAB_SIZE = len(review_dict)
    NUM_LABELS = 5

    input_dim = VOCAB_SIZE
    hidden_dim = 500
    output_dim = 5
    num_epochs = 100

    ff_nn_bow_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    ff_nn_bow_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ff_nn_bow_model.parameters(), lr=0.001)

    # Open the file for writing loss
    ffnn_loss_file_name = 'ffnn_bow_class_big_loss_500_epoch_100_less_lr.csv'
    f = open(ffnn_loss_file_name, 'w')
    f.write('iter, loss')
    f.write('\n')
    losses = []
    iter = 0

    for epoch in range(num_epochs):
        if (epoch + 1) % 25 == 0:
            print("Epoch completed: " + str(epoch + 1))
        train_loss = 0
        for index, row in train_X.iterrows():
            # Clearing the accumulated gradients
            optimizer.zero_grad()

            # Make the bag of words vectors for stemmed tokens
            bow_vec = make_bow_vector(review_dict, row['tokenized_text'], VOCAB_SIZE)

            # Forward pass to get output
            probs = ff_nn_bow_model(bow_vec)

            # Get the target label
            target = make_target(train_Y[index])

            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_function(probs, target)
            # Accumulating the loss over time
            train_loss += loss.item()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
        f.write(str((epoch + 1)) + "," + str(train_loss / len(train_X)))
        f.write('\n')
        train_loss = 0
    f.close()

    from sklearn.metrics import classification_report
    bow_ff_nn_predictions = []
    original_lables_ff_bow = []
    with torch.no_grad():
        for index, row in test_X.iterrows():
            bow_vec = make_bow_vector(review_dict, row['tokenized_text'], VOCAB_SIZE)
            probs = ff_nn_bow_model(bow_vec)
            bow_ff_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
            original_lables_ff_bow.append(make_target(test_Y[index]).cpu().numpy()[0])
    print(classification_report(original_lables_ff_bow, bow_ff_nn_predictions))
    ffnn_loss_df = pd.read_csv(ffnn_loss_file_name)
    print(len(ffnn_loss_df))
    print(ffnn_loss_df.columns)
    ffnn_plt_500_padding_100_epochs = ffnn_loss_df[' loss'].plot()
    fig = ffnn_plt_500_padding_100_epochs.get_figure()
    fig.savefig("ffnn_bow_loss_500_padding_100_epochs_less_lr.pdf")


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Linear function 1: vocab_size --> 500
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 500 --> 500
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3 (readout): 500 --> 3
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3 (readout)
        out = self.fc3(out)

        return F.softmax(out, dim=1)


# Function to return the dictionary either with padding word or without padding
def make_dict(data):

    print("Dictionary without padding")
    review_dict = corpora.Dictionary(data['tokenized_text'])
    return review_dict


# Function to make bow vector to be used as input to network
def make_bow_vector(review_dict, sentence, VOCAB_SIZE):
    vec = torch.zeros(VOCAB_SIZE, dtype=torch.float64, device=device)
    for word in sentence:
        vec[review_dict.token2id[word]] += 1
    return vec.view(1, -1).float()


def make_target(label):
    label = int(label)
    label-=1
    return torch.tensor([label], dtype=torch.long, device=device)
