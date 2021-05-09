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
        if disp_text: print('created model.', end=' ')
    else:
        model = load_model()
        if not model:
            save_model(make_model())
            model = load_model()
            if disp_text: print('created model.', end=' ')
        else:
            if disp_text: print('loaded model.', end=' ')

    data = load_data()
    data, data_dev = split_data(data)

    data = [d for i,d in enumerate(data) if i in [8,10,13,14]]
    print()
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
        # one_batch = True
    elif config.batch_size < 1:
        config.batch_size = int(len(data)*config.batch_size)
        # one_batch = False
    # else: one_batch = False

    if disp_text: print(f'hm data: {len(data)}, hm dev: {len(data_dev)}, bs: {config.batch_size}, lr: {config.learning_rate}, \ntraining started @ {now()}')

    # data_losss, dev_losss = [], []
    # if not one_batch:
    #     if not config.all_losses: config.all_losses.append(dev_loss(model, data))
    #     data_losss.append(config.all_losses[-1])
    # if config.dev_ratio:
    #     dev_losss.append(dev_loss(model, data_dev))

    # if data_losss or dev_losss:
    #     if disp_text: print(f'initial loss(es): {data_losss[-1] if data_losss else ""} {dev_losss[-1] if dev_losss else ""}')

    for ep in range(config.hm_epochs):

        loss = 0

        for i, batch in enumerate(batchify_data(data)):

            loss += train_on(model, batch)

        # loss /= len(data)

        # if not one_batch: loss = dev_loss(model, data)
        # data_losss.append(loss)
        # config.all_losses.append(loss)
        # if config.dev_ratio: dev_losss.append(dev_loss(model, data_dev))

        if disp_text: print(f'epoch {ep}, loss {loss}, completed @ {now()}', flush=True)
        # if disp_text: print(f'epoch {ep}, loss {loss}, dev loss {dev_losss[-1] if config.dev_ratio else ""}, completed @ {now()}', flush=True)
        # if config.ckp_per_ep and ((ep+1)%config.ckp_per_ep==0):
        #         save_model(model,config.model_path+f'_ckp{ep}')

    # if one_batch: data_losss.append(dev_loss(model, data))

    # if disp_text: print(f'training ended @ {now()} \nfinal losses: {data_losss[-1]}, {dev_losss[-1] if config.dev_ratio else ""}', flush=True)
    # show(plot(data_losss))
    # if config.dev_ratio:
    #     show(plot(dev_losss))
    # if not config.fresh_model: show(plot(config.all_losses))

    return model # , [data_losss, dev_losss]


# def dev_loss(model, batch):
#     pass
#     # with no_grad():
#     #     loss = train_on(model, batch)
#     # return loss /len(batch)


##


if __name__ == '__main__':
    model = main()
    save_model(model, config.model_path+'_final')