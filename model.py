import config
from ext import pickle_save, pickle_load

from torch import              \
    (tensor, Tensor,
    zeros, ones, eye, randn,
    cat, stack, transpose,
    sigmoid, tanh, relu, softmax,
    pow, sqrt,
    abs, sum, norm, mean, min, max,
    float32, no_grad)
from torch.nn.init import xavier_normal_

##


def make_model():

    w0 = randn(config.in_size+config.state_size,config.state_size, requires_grad=False, dtype=float32)
    w1 = randn(config.in_size+config.state_size,config.out_size, requires_grad=False, dtype=float32)

    return [[w0],[w1]]


def prop_model(model, inp, layer=None, do_negative=False):

    with no_grad():

        if layer is None:
            state = sigmoid(inp @ model[0][0])
            out = sigmoid(inp @ model[1][0])
            return state, out

        else:
            result = sigmoid(inp @ model[layer][0])

            if do_negative:
                inp_neg = sigmoid(result @ transpose(model[layer][0],0,1))
                result_neg = sigmoid(inp_neg @ model[layer][0])
                return result, inp_neg, result_neg
            else: return result


def empty_state(batch_size=1):
    return zeros(batch_size,config.state_size) if not config.use_gpu else zeros(batch_size, config.state_size).cuda()


##


def train_on(model, sequences, init_state=None):

    init_state = empty_state(len(sequences)) if not init_state else init_state


    print('-- training L0 --')


    for t in range(config.max_seq_len):
        print(f't: {t}')

        for i in range(config.hm_epochs_per_t):
            disp_text = i%(config.hm_epochs_per_t//10)==0
            disp_losses = []

            model[0][0].grad = zeros(model[0][0].size())
            state = init_state

            for tt in range(t+1):

                inp = cat([stack([sequence[tt] for sequence in sequences],0),state],-1)
                state, inp_neg, state_neg = prop_model(model, inp, layer=0, do_negative=True)

                pos_grad = (transpose(inp.unsqueeze(1), 1, 2) * state.unsqueeze(1)).sum(0)
                neg_grad = (transpose(inp_neg.unsqueeze(1), 1, 2) * state_neg.unsqueeze(1)).sum(0)
                model[0][0].grad += neg_grad-pos_grad

                if disp_text: disp_losses.append(float(sum(abs(inp-inp_neg))))
            if disp_text: print(f'\tloss_{i}: {disp_losses}')

            model[0][0].grad /= (t+1)
            sgd(model) if config.optimizer == 'sgd' else adaptive_sgd(model)
            # if disp_text: print(f'\tweight: {max(abs(model[0][0]))}, {mean(abs(model[0][0]))}')


    print('-- training L1 --')


    states = [init_state]
    inps = []
    for t in range(config.max_seq_len-1):
        inps.append(cat([stack([sequence[t] for sequence in sequences],0),states[-1]],-1))
        states.append(prop_model(model, inps[-1], layer=0))
    inps = cat(inps,0)

    lbls = []
    for t in range(1,config.max_seq_len):
        lbls.append(stack([sequence[t] for sequence in sequences],0))
    lbls = cat(lbls,0)

    for i in range(config.hm_epochs_per_t):
        disp_text = i%(config.hm_epochs_per_t//10)==0

        outs = prop_model(model, inps, layer=1)

        loss = lbls-outs
        pos_grad = (transpose(inps.unsqueeze(1),1,2) * loss.unsqueeze(1)).sum(0)
        model[1][0].grad = -pos_grad /(config.max_seq_len-1)

        if disp_text: print(f'\tloss_{i}: {float(sum(abs(loss)))}')

        sgd(model) if config.optimizer == 'sgd' else adaptive_sgd(model)


 ##


def respond_to(model, sequence, state=None):

    state = empty_state(1) if not state else state

    for timestep in sequence:

        inp = cat([timestep.unsqueeze(0),state], -1)
        state, out = prop_model(model, inp)

    response = [out]

    for t in range(config.hm_extra_steps-1):

        inp = cat([out,state], -1)
        state, out = prop_model(model, inp)
        response.append(out)

    return response


##


def sgd(model, lr=None, batch_size=None):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    with no_grad():
        for layer in model:
            for param in layer:
                if param.grad is not None:
                    param -= lr * param.grad / batch_size
                    param.grad = None


moments, variances = [], []

def adaptive_sgd(model, lr=None, batch_size=None,
                 alpha_moment=0.9, alpha_variance=0.999, epsilon=1e-8,
                 do_moments=True, do_variances=True, ep_nr=0):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    global moments, variances
    if do_moments and not moments: moments = [[zeros(weight.size(), dtype=float32) if not config.use_gpu else zeros(weight.size(), dtype=float32).cuda() for weight in layer] for layer in model]
    if do_variances and not variances: variances = [[zeros(weight.size(), dtype=float32) if not config.use_gpu else zeros(weight.size(), dtype=float32).cuda() for weight in layer] for layer in model]

    with no_grad():
            for _, layer in enumerate(model):
                for __, param in enumerate(layer):
                    if param.grad is not None:

                        param.grad /= batch_size

                        if do_moments:
                            moments[_][__] = alpha_moment * moments[_][__] + (1-alpha_moment) * param.grad
                            moment_hat = moments[_][__] / (1-alpha_moment**(ep_nr+1))
                        if do_variances:
                            variances[_][__] = alpha_variance * variances[_][__] + (1-alpha_variance) * param.grad**2
                            variance_hat = variances[_][__] / (1-alpha_variance**(ep_nr+1))

                        param -= lr * (moment_hat if do_moments else param.grad) / ((sqrt(variance_hat)+epsilon) if do_variances else 1)
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
        global moments, variances
        if fresh_meta:
            moments, variances = [], []
        else:
            moments, variances = meta
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
