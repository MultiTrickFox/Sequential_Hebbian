
## data params

combine_instrus = True
min_octave = 3
max_octave = 5
multi_octave = True
polyphony = False
monophony_mode = 'h' # 'l':lowest, 'h':highest, 'm':most

beat_resolution = 4

data_path = 'data'
dev_ratio = 0


## model params

timestep_size = 12+1 if not multi_octave else 12*(max_octave-min_octave+1)+1

in_size = timestep_size
state_size = 2**4
out_size = timestep_size

act_fn = 's'

init_xavier = False


## train params

loss_squared = False

learning_rate = 1e-2

max_seq_len = 10

batch_size = 0
gradient_clip = 0
hm_epochs = 1
max_epochs_per_t = 1000
optimizer = 'custom'

model_path = 'models/model'
fresh_model = True
fresh_meta = True
ckp_per_ep = hm_epochs//10

use_gpu = False

all_losses = []


## interact params

pick_threshold = .2
hm_extra_steps = 100
hm_output_file = 1
output_file = 'resp'


##

config_to_save = ['beat_resolution', 'min_octave', 'max_octave', 'multi_octave', 'polyphony',
                  'timestep_size', 'in_size', 'out_size',
                  'state_size', 'act_fn',
                  'all_losses',
                 ]

