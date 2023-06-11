""" This module prepares midi file data and feeds it to the neural network for training """

import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, BatchNormalization as BatchNorm, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names and instruments
    pitchnames = sorted(set(item for item in notes))
    instruments = sorted(set(item.split("|")[1] for item in notes))
    n_vocab = len(pitchnames)
    n_instruments = len(instruments)

    network_input, network_output = prepare_sequences(notes, pitchnames, instruments)

    model = create_network(network_input, n_vocab, n_instruments)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        for part in midi.parts:
            notes_to_parse = None

            try:
                notes_to_parse = part.recurse()
            except:
                notes_to_parse = part.flat.notes

            part_name = part.partName if part.partName is not None else 'Unknown'  # 악기 이름이 없을 경우 'Unknown'으로 설정

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    pitch_name = str(element.pitch)
                    notes.append(pitch_name + "|" + part_name)  # 악기 식별자 추가
                elif isinstance(element, chord.Chord):
                    chord_notes = '.'.join(str(n) for n in element.normalOrder)
                    notes.append(chord_notes + "|" + part_name)  # 악기 식별자 추가

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, pitchnames, instruments):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # create a dictionary to map pitches and instruments to integers
    pitchnames_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    instruments_to_int = dict((inst, number) for number, inst in enumerate(instruments))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        input_sequence = [pitchnames_to_int[char.split("|")[0]] for char in sequence_in if char.split("|")[0] in pitchnames_to_int]
        output_sequence = instruments_to_int[sequence_out.split("|")[1]] if sequence_out.split("|")[1] in instruments_to_int else None

        if input_sequence and output_sequence is not None:
            network_input.append(input_sequence)
            network_output.append(output_sequence)

    n_patterns = len(network_input)

    if n_patterns == 0:
        raise ValueError("Not enough valid data to create input sequences.")

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(len(pitchnames))

    # convert the output to one-hot encoding
    network_output = np_utils.to_categorical(network_output, num_classes=len(instruments))

    return (network_input, network_output)




def create_network(network_input, n_vocab, n_instruments):
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
    model.add(Dense(n_instruments))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'

    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
