""" This module generates a MIDI file with piano and violin sounds
    using the trained neural networks """
import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization as BatchNorm, Activation

def generate():
    # Load the trained models for piano and violin
    # Name of hdf5 (Piano,Violon)
    piano_model = load_model('weights-improvement-piano-01.hdf5')
    violin_model = load_model('weights-improvement-violin-01.hdf5')

    # Load the notes used to train the models
    with open('data/piano_notes', 'rb') as filepath:
        piano_notes = pickle.load(filepath)
    with open('data/violin_notes', 'rb') as filepath:
        violin_notes = pickle.load(filepath)

    # Get all pitch names for piano and violin
    piano_pitchnames = sorted(set(item for item in piano_notes))
    violin_pitchnames = sorted(set(item for item in violin_notes))

    # Get the number of unique pitches for piano and violin
    piano_vocab = len(set(piano_notes))
    violin_vocab = len(set(violin_notes))

    # Prepare input sequences for piano and violin
    piano_network_input, piano_normalized_input = prepare_sequences(piano_notes, piano_pitchnames, piano_vocab)
    violin_network_input, violin_normalized_input = prepare_sequences(violin_notes, violin_pitchnames, violin_vocab)

    # Generate notes for piano and violin
    piano_prediction_output = generate_notes(piano_model, piano_network_input, piano_pitchnames, piano_vocab)
    violin_prediction_output = generate_notes(violin_model, violin_network_input, violin_pitchnames, violin_vocab)

    # Create a stream with piano and violin notes
    midi_stream = create_midi(piano_prediction_output, violin_prediction_output)

    # Write the MIDI file
    midi_stream.write('midi', fp='output.mid')

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # Map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    normalized_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        normalized_input.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(normalized_input, (n_patterns, 1))
    # Normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def load_model(model_path):
    """ Load the trained model """
    model = Sequential()
    model.add(LSTM(512, input_shape=(100, 1), recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights
    model.load_weights(model_path)

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # Pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # Generate notes
    for _ in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output

def create_midi(piano_notes, violin_notes):
    """ Create a MIDI stream with piano and violin notes """
    offset = 0
    output_notes = []

    # Create piano notes
    for pattern in piano_notes:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset for piano notes
        offset += 0.5

    # Create violin notes
    for pattern in violin_notes:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Violin()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Violin()
            output_notes.append(new_note)

        # Increase offset for violin notes
        offset += 0.5

    # Create a MIDI stream
    midi_stream = stream.Stream(output_notes)

    return midi_stream

if __name__ == '__main__':
    generate()
