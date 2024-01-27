from gensim.models import KeyedVectors
import numpy as np


# Path to the pre-trained model
MODEL_PATH = 'C:/Users/Yanis/Downloads/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'


def get_word_embedding(word):
    
    """
    Retrieves the word embedding vector for a given word.

    Parameters:
    word (str): The word to retrieve the embedding for.

    Returns:
    numpy.ndarray or None: The word embedding vector if the word is in the vocabulary,
    a vector of zeros if the word is '<PAD>', a random vector if the word is '<UNK>',
    or None if the word is not found in the vocabulary.
    """
    
    # Load Word2Vec model. Here, we are using a pre-trained model by Google
    # Make sure you have the model file in your directory or adjust the path accordingly
    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
    
    # Check if the word is in the vocabulary
    if word in model.key_to_index:
        return model[word]
    elif word == '<PAD>':
        return np.zeros((model.vector_size,)) # Return a vector of zeros if it is padding
    elif word == '<UNK>':
        return np.random.uniform(-0.01, 0.01, (model.vector_size,)) # Return a random vector if it is unknown
    else:
        return None

if __name__ == '__main__':
    word = 'cacaprout'
    print("Embedding for {} is {}".format(word, get_word_embedding(word)))
    print('Embedding size is {}'.format(get_word_embedding(word).shape))

    word='hello'
    print("Embedding for {} is {}".format(word, get_word_embedding(word)))
    print('Embedding size is {}'.format(get_word_embedding(word).shape))

