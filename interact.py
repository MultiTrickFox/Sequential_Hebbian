def main():

    import config

    from model import load_model
    model = load_model(config.model_path+'_final')
    while not model:
        config.model_path = input('valid model: ')
        model = load_model()

    from data import load_data, split_data
    d = load_data()
    d, _ = split_data(d)

    # from random import shuffle
    # shuffle(d)
    #d = d[:config.hm_output_file]
    d = [d[14]]

    for i,seq in enumerate(d):

        from model import respond_to
        seq = respond_to(model, seq)
        seq = [t.detach() for t in seq]
        if config.use_gpu:
            seq = [t.cpu() for t in seq]
        seq = [t.numpy() for t in seq]

        from data import note_reverse_dict, convert_to_midi
        seq_converted = []
        for timestep in seq:
            if config.act_fn=='t': timestep = (timestep+1)/2
            t_converted = ''
            for i,e in enumerate(timestep[0]):
                if e>config.pick_threshold:
                    t_converted += note_reverse_dict[i%12]+str(int(i/12)+config.min_octave) if i!=config.out_size-1 else 'R'
                    t_converted += ','
            t_converted = t_converted[:-1] if len(t_converted) else 'R'
            seq_converted.append(t_converted)
        convert_to_midi(seq_converted).show()

if __name__ == '__main__':
    main()