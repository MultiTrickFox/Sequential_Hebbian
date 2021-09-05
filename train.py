import config
from ext import now
from model import make_model, train_on
from model import load_model, save_model
from data import load_data, split_data, batchify_data

##


def main(disp_text=True):

    if config.fresh_model:
        config.all_losses = []
        save_model(make_model())
        model = load_model()
        if disp_text: print('created model.')
    else:
        model = load_model()
        if not model:
            save_model(make_model())
            model = load_model()
            if disp_text: print('created model.')
        else:
            if disp_text: print('loaded model.')

    data = load_data()
    data, data_dev = split_data(data)

    data = [d for i,d in enumerate(data) if i in [8,10,13,14]]
    seq_lens = [len(d) for d in data]
    print(f'seq lens: {seq_lens}')
    min_seq_len = min(seq_lens)
    print(f'min seq len: {min_seq_len}')
    if not config.max_seq_len or config.max_seq_len > min_seq_len:
        config.max_seq_len = min_seq_len
    data = [d[:config.max_seq_len] for d in data]

    # from random import choice
    # from torch import randn
    # data = [[randn(config.in_size) for _ in range(choice(range(config.max_seq_len//2,config.max_seq_len)))] for _ in range(10)]
    # data_dev = []
    # for d in data: print(len(d))

    if not config.batch_size or config.batch_size >= len(data):
        config.batch_size = len(data)
    elif config.batch_size < 1:
        config.batch_size = int(len(data)*config.batch_size)

    if disp_text: print(f'hm data: {len(data)}, hm dev: {len(data_dev)}, bs: {config.batch_size}, lr: {config.learning_rate}, \ntraining started @ {now()}')

    for ep in range(config.hm_epochs):

        for i, batch in enumerate(batchify_data(data)):

            train_on(model, batch)

    return model


##


if __name__ == '__main__':
    model = main()
    save_model(model, config.model_path+'_final')