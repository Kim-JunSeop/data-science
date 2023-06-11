""" This module generates notes for a midi file using the
    trained neural network """
import pickle

import music21
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization as BatchNorm, Dropout, Activation
from keras.utils import np_utils


def generate():
    """ Generate a midi file with various instruments """
    # Load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input, instruments = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, instruments)
    create_midi(prediction_output)


def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # Map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    instruments = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])
        if sequence_out.startswith('_'):
            instrument_name = sequence_out.split(' ')[1]
            instruments.append(instrument_name)
        else:
            instruments.append(None)

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    normalized_input = normalized_input / float(n_vocab)

    return network_input, normalized_input, instruments


def create_network(network_input, n_vocab):
    """ Create the structure of the neural network """
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
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights('weights-improvement-02-6.5766-bigger.hdf5')

    return model


def generate_notes(model, network_input, pitchnames, n_vocab, instruments):
    """ Generate notes from the neural network based on a sequence of notes """
    start = numpy.random.randint(0, len(network_input) - 1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # Generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    # Combine prediction output with instrument information
    combined_output = []
    for note, instrument_name in zip(prediction_output, instruments):
        if instrument_name:
            combined_output.append('_' + instrument_name)
        combined_output.append(note)

    return combined_output


def create_midi(prediction_output):
    """ Convert the output from the prediction to notes and create a MIDI file from the notes """
    offset = 0
    output_notes = []

    def get_instrument(instrument_name):
        """ Get the instrument object based on the instrument name """
        if instrument_name == 'Piano':
            return instrument.Piano()
        elif instrument_name == 'Clavichord':
            return instrument.Clavichord()
        elif instrument_name == 'Sampler':
            return instrument.Sampler()
        elif instrument_name == 'StringInstrument':
            return instrument.StringInstrument()
        elif instrument_name == 'Brass':
            return instrument.BrassInstrument()
        elif instrument_name == 'Electric Guitar':
            return instrument.ElectricGuitar()
        elif instrument_name == 'Electric Bass':
            return instrument.ElectricBass()
        else:
            return instrument.Piano()

    # Create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # Check if the pattern starts with an underscore (indicating an instrument change)
        if pattern.startswith('_'):
            instrument_name = pattern.split(' ')[1]  # Extract instrument name
            instrument_obj = get_instrument(instrument_name)  # Get instrument object

            new_note = note.Rest()
            new_note.offset = offset
            new_note.storedInstrument = instrument_obj
            output_notes.append(new_note)

        else:
            # Check if the pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    try:
                        new_note = note.Note(int(current_note))
                        new_note.storedInstrument = instrument.Piano()
                        notes.append(new_note)
                    except ValueError:
                        continue
                if len(notes) > 0:
                    new_chord = chord.Chord(notes)
                    new_chord.offset = offset
                    output_notes.append(new_chord)
            else:
                try:
                    new_note = note.Note(pattern)
                except music21.pitch.AccidentalException:
                    continue
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

        # Increase offset each iteration so that notes do not stack
        offset += 1

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')


if __name__ == '__main__':
    generate()
