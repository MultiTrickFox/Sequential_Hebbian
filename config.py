
## data params

    ## pre process
beat_resolution = 4
combine_instrus = True
min_octave = 3
max_octave = 5
    ## post process
multi_octave = True
polyphony = False
monophony_mode = 'h' # 'l':lowest, 'h':highest, 'm':most

data_path = 'data'
dev_ratio = 0


## model params

timestep_size = 12+1 if not multi_octave else 12*(max_octave-min_octave+1)+1

in_size = timestep_size
state_size = 2**5
out_size = timestep_size


## train params

learning_rate = 1e-1

max_seq_len = 20

batch_size = 0
hm_epochs = 1
hm_epochs_per_t = 1000
optimizer = 'adaptive'

model_path = 'models/model'
fresh_model = True
fresh_meta = True

use_gpu = False


## interact params

pick_threshold = 1/((max_octave-min_octave+1)*12+1) if multi_octave else 1/13
hm_extra_steps = beat_resolution*4 *2
hm_output_file = 1


##

config_to_save = ['beat_resolution', 'min_octave', 'max_octave', 'multi_octave', 'polyphony',
                  'timestep_size', 'in_size', 'out_size', 'state_size', 'act_fn',
                 ]

