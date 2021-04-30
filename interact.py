def main():

    import config

    from model import load_model
    model = load_model()
    while not model:
        config.model_path = input('valid model: ')
        model = load_model()

    from data import load_data, split_data
    d = load_data()
    d, _ = split_data(d)

    # from random import shuffle
    # shuffle(d)
    d = d[:config.hm_output_file]

    for i,seq in enumerate(d):

        from model import respond_to
        seq = respond_to(model, [seq], training_run=False, extra_steps=config.hm_extra_steps)
        seq = [t.detach() for t in seq]
        if config.use_gpu:
            seq = [t.cpu() for t in seq]
        seq = [t.numpy() for t in seq]

        from data import note_reverse_dict, convert_to_stream
        seq_converted = []
        for t in seq:
            t_converted = ''
            for i,e in enumerate(t[0]):
                if e>config.pick_threshold:
                    t_converted += note_reverse_dict[i]+','
            t_converted = t_converted[:-1]
            seq_converted.append(t_converted)
        convert_to_stream(seq_converted).show()

if __name__ == '__main__':
    main()