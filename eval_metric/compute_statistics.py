import numpy as np
import music21 as m21
import os
import pandas as pd
import gc

def featurize(src):
  pitch = []
  for note in src.flat.notes:
    if note.isChord:
      for _note in note._notes:
        pitch.append(_note.pitch.pitchClass + _note.pitch.octave * 12)
    else:
      pitch.append(note.pitch.pitchClass + note.pitch.octave * 12)

  pitchCount = np.unique(np.array(pitch)).size

  pitchClasses = np.array(pitch) % 12

  pitch_class_hist = np.zeros((12)) #
  values, pitch_class_counts = np.unique(pitchClasses, return_counts=True)

  pitch_class_hist[values] = pitch_class_counts

  pitch_class_trans = np.zeros((12,12))
  values, pitch_class_counts = np.unique(np.stack([pitchClasses[:-1], pitchClasses[1:]]), axis=1, return_counts=True)
  pitch_class_trans[values[0], values[1]] = pitch_class_counts

  pitchRange = np.max(pitch) - np.min(pitch)

  avgPitchInterval = np.mean(np.array(pitch[1:]) - np.array(pitch[:-1]))

  offsets = np.array([note.offset for note in src.flat.notes])
  durations = np.array([note.duration.quarterLength for note in src.flat.notes]).astype(np.float32)

  avg_ioi = np.mean(np.diff(offsets))

  duration_accepted_values = [
      0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.,
      2.25, 2.5, 2.75, 3., 3.5, 3.75, 4.
  ]
  duration_map_to_idx = {val: ii for ii, val in enumerate(duration_accepted_values)}
  vectorized = np.vectorize(lambda x: duration_map_to_idx[x] if x in duration_map_to_idx else 15)


  duration_hist = np.zeros((16))
  values, duration_counts = np.unique(durations, return_counts=True)
  values = vectorized(values)
  duration_hist[values] = duration_counts
  duration_hist = duration_hist[:15]

  duration_trans_matrix = np.zeros((16,16))
  values, duration_trans_counts = np.unique(np.stack([durations[:-1], durations[1:]]), axis=1, return_counts=True)
  values = vectorized(values)
  duration_trans_matrix[values[0], values[1]] = duration_trans_counts
  duration_trans_matrix = duration_trans_matrix[:15, :15]

  return np.concatenate((np.array([pitchCount, pitchRange, avgPitchInterval, len(src.flat.notes), avg_ioi]),
                    pitch_class_hist, pitch_class_trans.flatten(),
                    duration_hist, duration_trans_matrix.flatten()))

metadata = pd.read_csv('maestro-v2.0.0/maestro-v2.0.0.csv')

beethoven_files = metadata[metadata['canonical_composer'] == 'Ludwig van Beethoven']['midi_filename']
other_files = metadata[metadata['canonical_composer'] != 'Ludwig van Beethoven']['midi_filename']

src = {'beethoven': [], 'other': []}

file_size = 10
start_index = file_size
for start_index in range(0, len(beethoven_files), file_size):
    end_index = start_index + file_size
    for i, file in enumerate(beethoven_files[start_index:end_index]):
      print('{} / {}'.format(i + start_index, len(beethoven_files)))
      src['beethoven'].append(m21.converter.parse('maestro-v2.0.0/' + file))


    data_X = np.stack([featurize(stream) for stream in src['beethoven']])
    data_y = np.zeros((data_X.shape[0]))

    # data_X = np.concatenate([data_X, np.stack([featurize(stream) for stream in src['other']])])
    # data_y = np.concatenate([data_y, np.zeros((data_X.shape[0] - data_y.shape[0]))])

    np.save(open("data_x_%d.npy" % (start_index // file_size), "wb"), data_X)
    np.save(open("data_y_%d.npy" % (start_index // file_size), "wb"), data_y)
    src['beethoven'] = []
    gc.collect()
