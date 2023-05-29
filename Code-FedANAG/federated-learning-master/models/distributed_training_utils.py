import torch
import copy
import math
from torch import nn, autograd
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import Dataset
import torch.nn.functional as F

max_norm = 10

def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def add_mome(target, source, beta_):
    for name in target:
        target[name].data = (beta_ * target[name].data + source[name].data.clone())


def add_mome2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()


def add_mome3(target, source1, source2, source3, beta_1, beta_2, beta_3):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone() + beta_3 * source3[name].data.clone()

def add_2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data += beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()

def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def scale_ts(target, source, scaling):
    for name in target:
        target[name].data = scaling * source[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()


def computer_norm(source1, source2):
    diff_norm = 0

    for name in source1:
        diff_source = source1[name].data.clone() - source2[name].data.clone()
        diff_norm += torch.pow(torch.norm(diff_source),2)

    return (torch.pow(diff_norm, 0.5))

def majority_vote(target, sources, lr):
    for name in target:
        threshs = torch.stack([torch.max(source[name].data) for source in sources])
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def get_other_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name in exp_mdl:
            n_par += len(exp_mdl[name].data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name in mdl:
            temp = mdl[name].data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


class DistributedTrainingDevice(object):
    '''
  A distributed training device (Client or Server)
  data : a pytorch dataset consisting datapoints (x,y)
  model : a pytorch neural net f mapping x -> f(x)=y_
  hyperparameters : a python dict containing all hyperparameters
  '''

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

class Client(DistributedTrainingDevice):

    def __init__(self, model, args, trn_x, trn_y, dataset_name, id_num=0):
        super().__init__(model, args)

        self.trn_gen = DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                  batch_size=self.args.local_bs, shuffle=True)
        self.id = id_num
        self.local_epoch = int(np.ceil(trn_x.shape[0] / self.args.local_bs))
        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'scaffold':
            self.ci = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.delta_ci = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.c_plus = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.unbias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.dg = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'feddyn':
            self.histi = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.unbias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'fed_localnesterov' or self.args.method == 'fed_localnesterov_nocom' or self.args.method == 'fed_localmome':
            self.bias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.buffer_momentum = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.buffer_nestrov = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'fed_nesterov':
            self.bias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'mime':
            self.bias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.buffer_m = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        self.state_params_diff = 0.0
        self.train_loss = 0.0
        self.n_par = get_mdl_params([self.model]).shape[1]

    def synchronize_with_server(self, server):
        # W_client = W_server
        self.model = copy.deepcopy(server.model)
        self.W = {name: value for name, value in self.model.named_parameters()}

    def compute_bias(self, server):
        if self.args.method == 'fedprox':
            cld_mdl_param = torch.tensor(get_mdl_params([self.model], self.n_par)[0], dtype=torch.float32, device=self.args.device)
            self.state_params_diff = self.args.mu * (-cld_mdl_param)

        elif self.args.method == 'fed_nesterov':
            scale_ts(target=self.bias, source=server.v, scaling= - self.args.beta_ * self.args.beta_)
            self.state_params_diff = torch.tensor(get_other_params([self.bias], self.n_par)[0],
                                                  dtype=torch.float32,
                                                  device=self.args.device)
        elif self.args.method == 'mime':
            scale_ts(target=self.bias, source=server.buffer_m, scaling= 1)
            self.state_params_diff = torch.tensor(get_other_params([self.bias], self.n_par)[0],
                                                  dtype=torch.float32,
                                                  device=self.args.device)

        if self.args.method == 'feddyn':
            cld_mdl_param = torch.tensor(get_mdl_params([self.model], self.n_par)[0], dtype=torch.float32, device=self.args.device)
            hist_mdl_param = torch.tensor(get_other_params([self.histi], self.n_par)[0], dtype=torch.float32, device=self.args.device)
            self.state_params_diff = self.args.coe * (-cld_mdl_param + hist_mdl_param)

        if self.args.method == 'scaffold':
            for name in self.unbias:
                self.unbias[name].data = server.c[name].data.clone() - self.ci[name].data.clone()
            self.state_params_diff = torch.tensor(get_other_params([self.unbias], self.n_par)[0], dtype=torch.float32,
                                                  device=self.args.device)


    def mime_mome_train(self):
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, momentum=0,
                                        weight_decay=0)

        self.buffer_m = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        trn_gen_iter = self.trn_gen.__iter__()

        batch_loss = []
        for i in range(self.local_epoch):
            images, labels = trn_gen_iter.__next__()
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            optimizer.zero_grad()
            log_probs = self.model(images)
            loss_f_i = self.loss_func(log_probs, labels.reshape(-1).long())

            loss_f_i.backward()
            self.gradient = {name: value.grad for name, value in self.model.named_parameters()}
            add(self.buffer_m, self.gradient)

        scale_ts(target=self.buffer_m, source=self.buffer_m, scaling=1 / self.local_epoch)
        optimizer.zero_grad()

    def train_cnn(self, server, iter):

        self.model.train()

        if self.args.method == 'fedavg' or self.args.method == 'fedavgm' or self.args.method == 'fedavg_gnag' or self.args.method == 'fed_nesterov' or self.args.method == 'mime' or self.args.method == 'scaffold':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay)

        if self.args.method == 'fedprox':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay + self.args.mu)

        if self.args.method == 'fed_localnesterov' or self.args.method == 'fed_localnesterov_nocom' or self.args.method == 'fed_localmome':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay)
            self.buffer_momentum = copy.deepcopy(server.momentum_buff)


        if self.args.method == 'feddyn':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay + self.args.coe)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

        # train and update
        epoch_loss = []
        for iter in range(self.args.local_ep):
            trn_gen_iter = self.trn_gen.__iter__()
            batch_loss = []
            for i in range(self.local_epoch):

                images, labels = trn_gen_iter.__next__()
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                optimizer.zero_grad()
                log_probs = self.model(images)
                loss_f_i = self.loss_func(log_probs, labels.reshape(-1).long())

                local_par_list = None
                for param in self.model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = torch.sum(local_par_list * self.state_params_diff)

                if self.args.method == 'fedavg' or self.args.method == 'fedavgm' or self.args.method == 'fedavg_gnag' or self.args.method == 'fed_localnesterov' or self.args.method =='fed_localnesterov_nocom' or self.args.method == 'fed_localmome':
                    loss = loss_f_i
                elif self.args.method == 'fedprox' or self.args.method == 'scaffold' or self.args.method == 'feddyn':
                    loss = loss_f_i + loss_algo
                elif self.args.method == 'fed_nesterov':
                    loss = (1 + self.args.beta_) * loss_f_i + loss_algo
                elif self.args.method == 'mime':
                    loss = (1 - self.args.beta_) * loss_f_i + self.args.beta_ * loss_algo
                    #loss = loss / (1-self.args.beta_** (iter+1))          #bias-corrected first moment estimate
                else:
                    exit('Error: unrecognized parameter')

                loss.backward()

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)

                if self.args.method == 'fed_localnesterov' or self.args.method == 'fed_localnesterov_nocom':
                    self.gradient = {name: value.grad for name, value in self.model.named_parameters()}
                    add_mome2(target=self.gradient, source1=self.gradient, source2=self.W,
                              beta_1=1, beta_2=self.args.weigh_delay)
                    add_mome(target=self.buffer_momentum, source=self.gradient, beta_=self.args.beta_)
                    add_mome2(target=self.buffer_nestrov, source1=self.buffer_momentum, source2=self.gradient, beta_1=self.args.beta_,
                              beta_2=1)
                    add_mome2(target = self.W, source1 = self.W, source2 = self.buffer_nestrov, beta_1 = 1, beta_2 = -self.args.lr)

                elif self.args.method == 'fed_localmome':
                    self.gradient = {name: value.grad for name, value in self.model.named_parameters()}
                    add_mome2(target=self.gradient, source1=self.gradient, source2=self.W,
                              beta_1=1, beta_2=self.args.weigh_delay)
                    add_mome(target=self.buffer_momentum, source=self.gradient, beta_=self.args.beta_)
                    add_mome2(target = self.W, source1 = self.W, source2 = self.buffer_momentum, beta_1 = 1, beta_2 = -self.args.lr)

                elif self.args.method == 'fedavg' or self.args.method == 'fedavgm' or self.args.method == 'fedavg_gnag' or self.args.method == 'fedprox' or self.args.method == 'fed_nesterov' or self.args.method == 'fed_nesterov' or self.args.method == 'scaffold' or self.args.method == 'feddyn' or self.args.method == 'mime':
                    optimizer.step()

                else:
                    exit('Error: unrecognized parameter')

                batch_loss.append(loss.item())

            scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def compute_weight_update(self, server, iter):

        if self.args.method == 'mime':
            self.mime_mome_train()

        # Training mode
        self.model.train()

        # W_old = W
        self.W_old = copy.deepcopy(self.W)

        # W = SGD(W, D)
        self.train_loss = self.train_cnn(server, iter)

        # dW = W - W_old
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        if self.args.method == 'scaffold':
            scale_ts(target=self.dg, source=self.dW,
                     scaling=(1 / (self.args.lr * self.local_epoch * self.args.local_ep)))

            add_mome3(target=self.c_plus, source1=self.ci, source2=server.c, source3=self.dg, beta_1=1,
                      beta_2=-1, beta_3=-1)
            add_mome2(target=self.delta_ci, source1=self.c_plus, source2=self.ci, beta_1=1, beta_2=-1)
            scale_ts(target=self.ci, source=self.c_plus, scaling=1)

        if self.args.method == 'feddyn':
            add(target=self.histi, source=self.dW)


class Server(DistributedTrainingDevice):

    def __init__(self, model, args):
        super().__init__(model, args)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_now = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.mome = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.gnag = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.all_model = copy.deepcopy(model).to(args.device)
        self.all_W = {name: value for name, value in self.all_model.named_parameters()}
        if self.args.method == 'fedavg_gnag':
            self.gnag = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        if self.args.method == 'fed_localnesterov' or self.args.method == 'fed_localmome' or self.args.method == 'fed_localnesterov_nocom':
            self.momentum_buff = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        if self.args.method == 'fed_nesterov':
            self.nesterov_buff = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.v = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        if self.args.method == 'mime':
            self.buffer_m_c = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.buffer_m = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        if self.args.method == 'scaffold':
            self.c = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.delta_c = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        if self.args.method == 'feddyn':
            self.hist = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.delta_dyn= {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.local_epoch = 0

    def aggregate_weight_updates(self, clients, iter, aggregation="mean"):

        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch
        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":
            average(target=self.dW, sources=[client.dW for client in clients])


    def computer_weight_update_down_dw(self, clients, iter):

        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch

        if self.args.method == 'fedavgm':
            add_mome(target = self.mome, source =self.dW, beta_ = self.args.beta_)
            add_mome2(target=self.W, source1=self.W, source2=self.mome, beta_1=1, beta_2=self.args.globallr)
        elif self.args.method == 'fedavg_gnag':
            add_mome(target = self.mome, source =self.dW, beta_ = self.args.beta_)
            add_mome2(target=self.gnag, source1=self.mome, source2=self.dW, beta_1=self.args.beta_, beta_2=1)
            add_mome2(target=self.W, source1=self.W, source2=self.gnag, beta_1=1, beta_2=self.args.globallr)
        else:
            add_mome2(target=self.W, source1=self.W, source2=self.dW, beta_1=1, beta_2=self.args.globallr)


        if self.args.method == 'scaffold':

            average(target=self.delta_c, sources=[client.delta_ci for client in clients])
            scale(target=self.delta_c, scaling=self.args.frac)
            add(target=self.c, source=self.delta_c)

        if self.args.method == 'feddyn':

            scale_ts(target=self.delta_dyn, source=self.dW, scaling=self.args.frac)
            add(target=self.hist, source=self.delta_dyn)
            add(target=self.W, source=self.hist)

        if self.args.method == 'fed_localnesterov' or self.args.method == 'fed_localmome':
            average(target=self.momentum_buff, sources=[client.buffer_momentum for client in clients])

        if self.args.method == 'fed_localnesterov_nocom':
            self.momentum_buff = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}


        elif self.args.method == 'fed_nesterov':
            add_mome2(target=self.nesterov_buff, source1=self.dW, source2=self.v, beta_1=1,
                      beta_2= - self.local_epoch * self.args.local_ep * self.args.beta_*self.args.beta_ * self.args.lr)
            add_mome2(target=self.v, source1=self.v, source2=self.nesterov_buff, beta_1=self.args.beta_,
                      beta_2=1/( self.local_epoch * self.args.local_ep * (1+self.args.beta_) * self.args.lr))

        elif self.args.method == 'mime':
            average(target=self.buffer_m_c, sources=[client.buffer_m for client in clients])
            add_mome2(target = self.buffer_m, source1=self.buffer_m_c, source2=self.buffer_m, beta_1 = 1-self.args.beta_, beta_2=self.args.beta_)


    @torch.no_grad()
    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.model(data)
            # sum up batch loss
            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()
            # get the index of the max log-probability
            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
            '''
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            '''
        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss

    @torch.no_grad()
    def all_model_evaluate(self, clients, data_x, data_y, dataset_name):

        average(target=self.all_W, sources=[client.W for client in clients])
        self.all_model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.all_model(data)
            # sum up batch loss
            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()
            # get the index of the max log-probability
            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
            '''
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            '''
        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss

