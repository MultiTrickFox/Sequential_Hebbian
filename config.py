
## data params

combine_instrus = True
polyphony = True

beat_resolution = 4

data_path = 'data'
dev_ratio = 0


## model params

timestep_size = 12

in_size = timestep_size
out_size = timestep_size

state_size = in_size//4

act_fn = 's'

init_xavier = False


## train params

loss_squared = False

learning_rate = 1e-1

max_seq_len = 0

batch_size = 0
gradient_clip = 0
hm_epochs = 100
hm_epochs_per_t = 100
optimizer = 'custom'

model_path = 'models/model'
fresh_model = True
fresh_meta = True
ckp_per_ep = hm_epochs//10

use_gpu = False

all_losses = []


## interact params

hm_extra_steps = 100

hm_output_file = 1

output_file = 'resp'


##

config_to_save = ['beat_resolution',
                  'state_size', 'act_fn',
                  'all_losses',
                 ]

