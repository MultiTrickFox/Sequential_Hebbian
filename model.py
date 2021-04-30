import config
from ext import pickle_save, pickle_load

from torch import              \
    (tensor, Tensor,
    zeros, ones, eye, randn,
    cat, stack, transpose,
    sigmoid, tanh, relu, softmax,
    pow, sqrt,
    abs, sum, norm, mean,
    float32, no_grad)
from torch.nn.init import xavier_normal_

##


def make_model():

    w = randn(config.in_size+config.state_size,config.state_size, requires_grad=False)
    return [[w]]

def prop_model(model, inp):

    act_fn = None if not config.act_fn else (sigmoid if config.act_fn=='s' else tanh)

    with no_grad():

        state = inp @ model[0][0]
        if act_fn: state = act_fn(state)

        inp_neg = state @ transpose(model[0][0], 0,1)
        if act_fn: inp_neg = act_fn(inp_neg)

        state_neg = inp_neg @ model[0][0]
        if act_fn: state_neg = act_fn(state_neg)

        return state, inp_neg, state_neg


def empty_state(batch_size=1):
    return zeros(batch_size,config.state_size) if not config.use_gpu else zeros(batch_size, config.state_size).cuda()


##


def respond_to(model, sequences, state=None, training_run=True, extra_steps=0):

    if training_run:

        loss = 0
        state = empty_state(len(sequences)) if not state else state

        max_seq_len = max(len(sequence) for sequence in sequences)
        has_remaining = list(range(len(sequences)))

        for t in range(max_seq_len):

            #print(f't: {t}')

            has_remaining = [i for i in has_remaining if len(sequences[i][t:t+1])]

            inp = stack([sequences[i][t] for i in has_remaining],0)
            partial_state = stack([state[i] for i in has_remaining],0)

            inp = cat([inp,partial_state], -1)

            for i in range(config.max_epochs_per_t):

                #print(f'\ti: {i}')

                partial_state, neg_inp, neg_partial_state = prop_model(model, inp)

                pos_grad = (transpose(inp.unsqueeze(1), 1,2) * partial_state.unsqueeze(1)).sum(0)
                neg_grad = (transpose(neg_inp.unsqueeze(1), 1,2) * neg_partial_state.unsqueeze(1)).sum(0)

                grad = -(pos_grad-neg_grad)
                loss_t_i = pow(inp-neg_inp,2) if config.loss_squared else abs(inp-neg_inp)

                # for i,l,g in zip(has_remaining,loss_t_i,grad):
                #     l /= len(sequences[i])
                #     g /= len(sequences[i])
                model[0][0].grad = grad
                # loss += sum(loss_t_i)/config.hm_epochs_per_t

                input(f'\tloss_t_i: {sum(loss_t_i)}')

                sgd(model) if config.optimizer == 'sgd' else adaptive_sgd(model)

            loss += sum(loss_t_i)

            print(f'loss_t: {sum(loss_t_i)}')

            for ii,i in enumerate(has_remaining):
                state[i] = partial_state[ii]

        return loss

    else:

        state = empty_state(1) if not state else state

        for timestep in sequences[0]:
            inp = cat([timestep.unsqueeze(0),state], -1)
            state, _,_ = prop_model(model, inp)

        response = []

        act_fn = None if not config.act_fn else (sigmoid if config.act_fn == 's' else tanh)

        for t in range(extra_steps):

            inp = state @ transpose(model[0][0], 0, 1)
            if act_fn: inp = act_fn(inp)

            out = inp[...,:config.in_size]
            state = inp[...,config.in_size:]

            response.append(out)

        return response


##


def sgd(model, lr=None, batch_size=None):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    with no_grad():

        for layer in model:
            for param in layer:

                param.grad /=batch_size

                if config.gradient_clip:
                    param.grad.clamp(min=-config.gradient_clip,max=config.gradient_clip)

                param -= lr * param.grad
                param.grad = None


moments, variances, ep_nr = [], [], 0

def adaptive_sgd(model, lr=None, batch_size=None,
                 alpha_moment=0.9, alpha_variance=0.999, epsilon=1e-8,
                 do_moments=True, do_variances=True):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    global moments, variances, ep_nr
    if not (moments or variances):
        if do_moments: moments = [[zeros(weight.size()) if not config.use_gpu else zeros(weight.size()).cuda() for weight in layer] for layer in model]
        if do_variances: variances = [[zeros(weight.size()) if not config.use_gpu else zeros(weight.size()).cuda() for weight in layer] for layer in model]

    ep_nr +=1

    with no_grad():
            for _, layer in enumerate(model):
                for __, param in enumerate(layer):

                    lr_ = lr
                    param.grad /= batch_size

                    if do_moments:
                        moments[_][__] = alpha_moment * moments[_][__] + (1-alpha_moment) * param.grad
                        moment_hat = moments[_][__] / (1-alpha_moment**(ep_nr+1))
                    if do_variances:
                        variances[_][__] = alpha_variance * variances[_][__] + (1-alpha_variance) * param.grad**2
                        variance_hat = variances[_][__] / (1-alpha_variance**(ep_nr+1))

                    param -= lr_ * (moment_hat if do_moments else param.grad) / ((sqrt(variance_hat)+epsilon) if do_variances else 1)
                    param.grad = None


##


def save_model(model, path=None):
    from warnings import filterwarnings
    filterwarnings("ignore")
    if not path: path = config.model_path
    if path[-3:]!='.pk': path+='.pk'
    if config.use_gpu:
        moments_ = [[e2.detach().cuda() for e2 in e1] for e1 in moments]
        variances_ = [[e2.detach().cuda() for e2 in e1] for e1 in variances]
        meta = [moments_, variances_]
        model = pull_copy_from_gpu(model)
    else:
        meta = [moments, variances]
    meta.append(ep_nr)
    configs = [[field,getattr(config,field)] for field in dir(config) if field in config.config_to_save]
    pickle_save([model,meta,configs],path)


def load_model(path=None, fresh_meta=None):
    if not path: path = config.model_path
    if not fresh_meta: fresh_meta = config.fresh_meta
    if path[-3:]!='.pk': path+='.pk'
    obj = pickle_load(path)
    if obj:
        model, meta, configs = obj
        if config.use_gpu:
            TorchModel(model).cuda()
        global moments, variances, ep_nr
        if fresh_meta:
            moments, variances, ep_nr = [], [], 0
        else:
            moments, variances, ep_nr = meta
            if config.use_gpu:
                moments = [[e2.cuda() for e2 in e1] for e1 in moments]
                variances = [[e2.cuda() for e2 in e1] for e1 in variances]
        for k_saved, v_saved in configs:
            v = getattr(config, k_saved)
            if v != v_saved:
                if v=='all_losses' and fresh_meta: continue
                print(f'config conflict resolution: {k_saved} {v} -> {v_saved}')
                setattr(config, k_saved, v_saved)
        return model


##


from torch.nn import Module, Parameter

class TorchModel(Module):

    def __init__(self, model):
        super(TorchModel, self).__init__()
        for layer_i, layer in enumerate(model):
            for param_i,param in enumerate(layer):
                if type(param) != Parameter:
                    param = Parameter(param)
                setattr(self,f'layer{layer_i}_param{param_i}',param)

            model[layer_i] = list(getattr(self, f'layer{layer_i}_param{param_i}') for param_i in range(len(model[layer_i])))
        self.model = model

    def forward(self, states, inp):
        prop_model(self.model, states, inp)


def pull_copy_from_gpu(model):
    return [list(weight.detach().cpu() for weight in layer) for layer in model]
