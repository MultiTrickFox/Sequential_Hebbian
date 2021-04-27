import config

from ext import pickle_save, pickle_load

from glob import glob

from pretty_midi import PrettyMIDI

from music21 import *

from torch import tensor, float32

from random import shuffle

from math import ceil

from copy import deepcopy

##


note_dict = {
    'A': 0,
    'A#': 1, 'B-': 1,
    'B': 2,
    'C': 3,
    'C#': 4, 'D-': 4,
    'D': 5,
    'D#': 6, 'E-': 6,
    'E': 7,
    'F': 8,
    'F#': 9, 'G-': 9,
    'G': 10,
    'G#': 11, 'A-': 11,
    'R': 12
}

note_reverse_dict = {
    0: 'A',
    1: 'A#',
    2: 'B',
    3: 'C',
    4: 'C#',
    5: 'D',
    6: 'D#',
    7: 'E',
    8: 'F',
    9: 'F#',
    10: 'G',
    11: 'G#',
    12: 'R'
}

empty_vector = [0 for _ in range(config.in_size)]


##


def preprocess():
    data = []

    raw_files = glob(config.data_path + "/**/*.mid*") + glob(config.data_path + "/**/*.MID*") + glob(
        config.data_path + "/*.mid*") + glob(config.data_path + "/*.MID*")

    print(f'files to read: {len(raw_files)}')

    for i, raw_file in enumerate(raw_files):

        try:
            data.extend(preprocess_file(raw_file))
        except Exception as e:
            print(f'ERROR: {raw_file} failed, {e}')

        if (i + 1) % 50 == 0:
            print(f'>> {i + 1}/{len(raw_files)}')

    print(f'>> obtained total of {len(data)} sequences.')

    return data


def preprocess_file(raw_file):
    print(f'> processing file {raw_file}')

    ## remove drums

    sound = PrettyMIDI(raw_file)
    drum_instruments_index = [i for i, inst in enumerate(sound.instruments) if inst.is_drum]
    for i in sorted(drum_instruments_index, reverse=True):
        del sound.instruments[i]
    sound.write(raw_file)

    ## read parts

    sample = converter.parse(raw_file)

    parts = instrument.partitionByInstrument(sample)
    if not parts:
        parts = [sample.flat]

    try:
        time_signatures = [int(part.timeSignature.ratioString[0]) for part in parts]
    except:
        print('WARNING: time signature failed, check the file')
        time_signatures = [4 for _ in range(len(parts))]

    ## convert parts

    converted_sequences = []

    for part, time_signature in zip(parts, time_signatures):

        converted_sequence = [[] for _ in range(len(part.makeMeasures()) * time_signature * config.beat_resolution)]

        for element in part.flat:

            try:
                assert element.beat
                assert element.duration

                add_to_sequence(converted_sequence, element)

            except:
                pass

        converted_sequences.append(converted_sequence)

    ## combine parts

    if config.combine_instrus:

        combined_converted_sequence = []

        max_len = max([len(part) for part in converted_sequences])

        for part in converted_sequences:
            if len(part) != max_len:
                for _ in range(max_len - len(part)):
                    part.append([])

        for t in range(max_len):
            t_collection = []
            for part in converted_sequences:
                t_collection.extend(part[t])
            combined_converted_sequence.append(t_collection)

        converted_sequences = [combined_converted_sequence]

    ## finalize

    converted_sequences = [[normalize_vector(vectorize_timestep(timestep)) for timestep in
                            trim_empty_timesteps(converted_sequence, time_signature)]
                           for converted_sequence, time_signature in zip(converted_sequences, time_signatures)]

    # for converted_sequence, time_signature in zip(converted_sequences, time_signatures):
    #     # if input('Show stream? (y/n): ').lower() == 'y':
    #     # from numpy.random import rand
    #     # if rand() <= .5:
    #         from interact import convert_to_stream
    #         sequence_string = [''.join(f'{note_reverse_dict[i]},' for i, element in enumerate(timestep) if element > 0)[:-1] for timestep in converted_sequence]
    #         convert_to_stream(sequence_string).show()
    #         input('Halt..')

    return zip(converted_sequences, time_signatures)


##


def add_to_sequence(converted_sequence, element):
    if isinstance(element, note.Note):
        vector = vectorize_element(element)
        starting_group = round(element.offset * config.beat_resolution)
        ending_group = int(element.duration.quarterLength * config.beat_resolution)
        if ending_group == 0: ending_group = 1
        for group in range(ending_group):
            converted_sequence[starting_group + group].append(vector)

    elif isinstance(element, chord.Chord):
        starting_group = round(
            element.offset * config.beat_resolution)  # normally this should've acted on "for each e in chord", thank you music21..
        ending_group = int(element.duration.quarterLength * config.beat_resolution)
        if ending_group == 0: ending_group = 1
        for e in element:
            vector = vectorize_element(e)
            for group in range(ending_group):
                converted_sequence[starting_group + group].append(vector)


##


def vectorize_element(element):
    vector = empty_vector.copy()
    vector[note_dict[element.pitch.name]] += 1

    return vector


def vectorize_timestep(timestep):
    vec = empty_vector.copy()

    if config.polyphony:

        for vector in timestep:
            for i, v in enumerate(vector):
                vec[i] += v

    else:

        counts = {}

        for vector in timestep:
            counts[str(vector)] = 0.0
        for vector in timestep:
            counts[str(vector)] += 1

        if counts:
            max_val = max(list(counts.values()))
            for k, v in counts:
                if v == max_val:
                    vec[k] += 1

    if vec == empty_vector:
        vec[-1] += 1

    return vec


def normalize_vector(vector):
    return [e / sum(vector) for e in vector]


##


def trim_empty_timesteps(converted_sequence, time_signature):
    trim_groups_of = time_signature * config.beat_resolution

    trim_from_start = 0
    for i in range(int(len(converted_sequence) / trim_groups_of)):
        if not any(converted_sequence[i * trim_groups_of + j] for j in range(trim_groups_of)):
            trim_from_start += 1
        else:
            break
    if trim_from_start:
        converted_sequence = converted_sequence[trim_from_start * trim_groups_of:]

    trim_from_end = 0
    for i in range(int(len(converted_sequence) / trim_groups_of) - 1, -1, -1):
        if not any(converted_sequence[i * trim_groups_of + j] for j in range(trim_groups_of)):
            trim_from_end += 1
        else:
            break
    if trim_from_end:
        converted_sequence = converted_sequence[:-trim_from_end * trim_groups_of]

    trim_from_mid = []
    for i in range(int(len(converted_sequence) / trim_groups_of)):
        if not any(converted_sequence[i * trim_groups_of + j] for j in range(trim_groups_of)):
            trim_from_mid.append(i)
    if trim_from_mid:
        converted_sequence_ = []
        for i in range(int(len(converted_sequence) / trim_groups_of)):
            if i not in trim_from_mid:
                converted_sequence_.extend(converted_sequence[i * trim_groups_of:(i + 1) * trim_groups_of])
        converted_sequence = converted_sequence_

    return converted_sequence


##


def save_data(data, path=None):
    if not path: path = config.data_path
    if path[-3:] != '.pk': path += '.pk'
    pickle_save(data, path)

def load_data(path=None):
    if not path: path = config.data_path
    if path[-3:] != '.pk': path += '.pk'
    data = pickle_load(path)
    if data:
        for i, (sequence, time_sig) in enumerate(data):
            data[i] = ([tensor(t,dtype=float32) for t in sequence]) #, time_sig)
            # todo : append to data the time_sig & current step ?
        return data


def split_data(data, dev_ratio=config.dev_ratio, do_shuffle=False):
    if do_shuffle: shuffle(data)
    hm_train = ceil(len(data) * (1 - dev_ratio))
    data_dev = data[hm_train:]
    data = data[:hm_train]
    return data, data_dev

def batchify_data(data, batch_size=config.batch_size, do_shuffle=True):
    if do_shuffle: shuffle(data)
    hm_batches = int(len(data) / batch_size)
    return [data[i * batch_size:(i + 1) * batch_size] for i in range(hm_batches)] \
        if hm_batches else [data]


##


def convert_to_stream(track):
    track = [timestep.split(',') for timestep in track]

    music_stream = stream.Stream()
    music_stream.timeSignature = meter.TimeSignature(f'4/4')
    music_stream.insert(0, metadata.Metadata(
        title='vanilla ai',
        composer=f'sent from {config.model_path}'))

    for i, timestep in enumerate(track):

        c = chord.Chord()

        for note_name in timestep:

            sustain = 1

            for other_timestep in track[i + 1:]:

                sustains_to_other = False

                for ii, other_note_name in enumerate(other_timestep):

                    if note_name == other_note_name:
                        sustains_to_other = True
                        sustain += 1
                        del other_timestep[ii]

                        if 'R' in other_timestep:
                            del other_timestep[other_timestep.index('R')]

                        break

                if not sustains_to_other:
                    break

            if note_name != 'R':
                n = note.Note(note_name + '4');
                n.duration.quarterLength *= sustain / config.beat_resolution
                c.add(n)
            else:
                n = note.Rest();
                n.duration.quarterLength *= sustain / config.beat_resolution

            # n.storedInstrument = instrument.Piano()
            n.offset = i / config.beat_resolution
            music_stream.append(n)
            n.offset = i / config.beat_resolution
            # n.storedInstrument = instrument.Piano()

    return music_stream


##


def main():
    save_data(preprocess())
    # path = config.data_path
    # if path[-3:] != '.pk': path += '.pk'
    # prev_data = pickle_load(path)
    # if prev_data: print('> appending data to prev file')
    # save_data(preprocess() + (prev_data if prev_data else []))


if __name__ == '__main__':
    main()
