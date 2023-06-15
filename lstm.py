import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, BatchNormalization as BatchNorm, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

def train_network():
    """ Train Neural Networks to generate music for each instrument """
    piano_notes, violin_notes = get_notes()
    # get amount of pitch names for each instrument
    piano_vocab = len(set(piano_notes))
    violin_vocab = len(set(violin_notes))

    piano_input, piano_output = prepare_sequences([piano_notes], piano_vocab)
    violin_input, violin_output = prepare_sequences([violin_notes], violin_vocab)

    piano_model = create_network(piano_input, piano_vocab)
    violin_model = create_network(violin_input, violin_vocab)

    train(piano_model, piano_input, piano_output, 'piano')
    train(violin_model, violin_input, violin_output, 'violin')


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    piano_notes = []
    violin_notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        parts = instrument.partitionByInstrument(midi)

        if parts:  # file has instrument parts
            for part in parts:
                instrument_name = part.partName.lower()
                if 'piano' in instrument_name:
                    notes_to_parse = part.recurse()
                    for element in notes_to_parse:
                        if isinstance(element, note.Note):
                            piano_notes.append(str(element.pitch))
                        elif isinstance(element, chord.Chord):
                            piano_notes.append('.'.join(str(n) for n in element.normalOrder))
                elif 'violin' in instrument_name:
                    notes_to_parse = part.recurse()
                    for element in notes_to_parse:
                        if isinstance(element, note.Note):
                            violin_notes.append(str(element.pitch))
                        elif isinstance(element, chord.Chord):
                            violin_notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/piano_notes', 'wb') as filepath:
        pickle.dump(piano_notes, filepath)

    with open('data/violin_notes', 'wb') as filepath:
        pickle.dump(violin_notes, filepath)

    print(piano_notes)
    print(violin_notes)
    input()

    return piano_notes, violin_notes


def prepare_sequences(notes, n_vocab):
    sequence_length = 500

    # get all pitch names
    pitchnames = sorted(set(item for sublist in notes for item in sublist))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for notes_instrument in notes:
        for i in range(0, len(notes_instrument) - sequence_length, 1):
            sequence_in = notes_instrument[i:i + sequence_length]
            sequence_out = notes_instrument[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    # convert the output to one-hot encoding
    network_output = np_utils.to_categorical(network_output, num_classes=n_vocab)
    print(network_input)
    print('-------------')
    print(network_output)
    input()

    return (network_input,network_output)



def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', run_eagerly=True)

    return model


def train(model, network_input, network_output, instrument_name):
    """ Train the neural network """
    filepath = f"weights-improvement-{instrument_name}-{{epoch:02d}}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    tf.config.run_functions_eagerly(True)
    model.fit(network_input, network_output, epochs=1, batch_size=32, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
