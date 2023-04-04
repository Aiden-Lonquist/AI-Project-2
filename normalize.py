import spacy

nlp = spacy.load('en_core_web_sm')

def normalize(text):
    # Tokenize the article: doc
    doc = nlp(text)

    # Lemmatize the tokens, convert to lowercase, and remove punctuation and stopwords
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

    # Replace numbers with NUM token
    tokens = [token if not token.isnumeric() else 'NUM' for token in tokens]

    # Put tokens back together and return it
    clean_text = ' '.join(tokens)
    return clean_text

