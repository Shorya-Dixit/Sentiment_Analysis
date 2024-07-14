import pickle
import keras
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def pad_sequences(sequences, padding='post', maxlen=None):
    """
    Pads sequences to the same length.

    Args:
        sequences: List of sequences to pad.
        padding: 'pre' or 'post' to pad sequences at the beginning or end.
        maxlen: Maximum length of sequences.

    Returns:
        A 2D NumPy array of padded sequences.
    """

    # Determine the maximum length of sequences
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    # Initialize a 2D NumPy array of zeros with shape (len(sequences), maxlen)
    padded_sequences = np.zeros((len(sequences), maxlen))

    # Pad the sequences
    for i, seq in enumerate(sequences):
        if padding == 'pre':
            padded_sequences[i, :len(seq)] = seq
        elif padding == 'post':
            padded_sequences[i, :len(seq)] = seq
            padded_sequences[i, len(seq):] = 0

    # Convert the padded sequences to integers
    padded_sequences = padded_sequences.astype(int)

    return padded_sequences

model = keras.models.load_model('model.h5')

with open('tokenizer.pickle', 'rb') as f:
    # Load the object from the pickle file
    tokenizer = pickle.load(f)


def predict_class(text,model,tokenizer):
    '''Function to predict sentiment class of the passed text'''

    sentiment_classes = ['negative', 'neutral', 'positive']
    max_len=50
    xt = tokenizer.texts_to_sequences(text)
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    yt = model.predict(xt).argmax(axis=1)
    return sentiment_classes[yt[0]]



