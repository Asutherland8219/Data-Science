from numpy.core.arrayprint import format_float_positional
import pandas as pd 
import numpy as np 
import kaggle

''' Note this needs installs before using '''

# pip install pytorch
# pip install fastai


path = untar_data(URLs.IMAGENETTE_160)

t = get_image_files(path)

# peek the image path and name 
print(t[0])

im = Image.open(files[0])
print(im)

im_t = tensor(im)
print(im_t.shape())

''' index the data set and create tensors of the images '''
class Dataset:
    def __init__(self, fns): self.fns=fns
    def __len__(self): return len(self.fns)
    def __getitem__(self, i):
        im = Image.open(self.fns[i]).resize((64, 64)).convert('RGB')
        y = v2i[self.fns[i].parent.name]
        return tensor(im).float()/255, tensor(y)
    
train_filt = L(o.parent.parent.name=='train' for o in files)
train, valid = files[train_filt], files[~train_filt]

print(len(train), len(valid))

train_ds, valid_ds = Dataset(train), Dataset(valid)
x, y = train_ds[0]

print(x.shape, y.shape)

''' Collate the variables into a mini batch '''

def collate(idxs, ds):
    xb, yb = zip(*[ds[i] for i in idxs])
    return torch.stack(xb), torch.stack(yb)

# peek the results 
x, y = collate([1,2], train_ds)

print(x.shape, y)

''' Create a data loader '''

class DataLoader:
    def __init__(self, ds, bs=128, shuffle=False, n_workers=1):
        self.ds, self.bs, self.shuffle, self.n_workers = ds, bs, shuffle, n_workers
    
    def __len__(self): return (len(self.ds)-1)//self.bs + 1

    def __iter__(self):
        idxs = L.range(self.ds)
        if self.shuffle: idxs = idxs.shuffle()
        chunks = [idxs[n:n + self.bs] for n in range(0, len(self.ds), self.bs)]
        with ProcessPoolExecutor(self.n_workers) as ex:
            yield from ex.map(collate, chunks, ds=self.ds)

# test it out on the new data set 

n_works = min(16, defaults.cpus)
train_dl = DataLoader(train_ds, bs=128, shuffle=True, n_workers=n_workers)
valid_dl = DataLoader(valid_ds, bs=256, shuffle=False,n_workers=n_workers)
xb, yb = first(train_dl)

print(xb.shape, yb.shape, len(train_dl))

''' Normalize Data '''

class Normalize:
    def __init__(self, stats): self.stats=stats
    def __call__(self, x):
        if x.device != self.stats[0].device:
            self.stats = to_device(self.stats, x.device)
        return (x-self.stats[0])/self.stats[1]


stats = [xb.mean((0,1,2)), xb.std((0,1,2))]
norm = Normalize(stats)
def tfm_x(x): return norm(x).permute((0,3,1,2))

t = tfm_x(x)
print(t.mean((0,2,3)), t.std(0,2,3))

''' Create a parameter class as a marker to show what is included in parameters '''

class Parameter(Tensor):
    def __new__(self, x): return Tensor._make_subclass(Parameter, x, True)
    def __init__(self, *args, **kwargs): self.requires_grad_()


''' Create Module '''

class Module:
    def __init__(self):
        self.hook, self.params, self.children, self._training = None, [], [], False
    
    def register_parameters(self, *ps): self.params += ps
    def register_modules (self, *ms): self.children += ms

    @property
    def training(self): return self._training

    @training.setter
    def training(self, v):
        self._training = v
        for m in self.children:
            m.training=v

    def parameters(self):
        return self.params + sum([m.parameters() for m in self.children], [])

    def __setattr__(self, k, v):
        super().__setattr__(k,v)
        if isinstance(v, Parameter): self.register_parameters(v)
        if isinstance(v, Module): self.register_modules(v)

    def __call__(self, *args, **kwargs):
        res = self.forward(*args, **kwargs)
        if self.hook is not None: self.hook(res, args)
        return res
    
    def cuda(self):
        for p in self.paramters(): p.data = p.data.cuda()
    
''' Create ConvLayer '''

class ConvLayer(Module):
    def __init__(self, ni, nf, stride=1, bias=True, act=True):
        super().__init__()
        self.w = Parameter(torch.zeros(nf, ni, 3, 3))
        self.b = Parameter(torch.zeros(nf)) if bias else None
        self.act, self.stride = act,stride
        init = nn.init.kaiming_normal_ if act else nn.init.xavier_normal_
        init(self.w)

    def forward(self, x):
        x = F.conv2d(x, self.w, self.b, stride=self.stride, padding=1)
        if self.act: x = F.relu(x)
        return x

''' Create a linear layer '''

class Linear(Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.w = Parameter(torch.zeros(nf, ni))
        self.b = Parameter(torch.zeros(nf))
        nn.init.xavier_normal_(self.w)

    def forward(self, x): return x @self.w.t() + self.b

''' Create Testing model '''
class T(Module):
    def __init__(self):
        super().__init__()
        self.c, self.l = ConvLayer(3, 4), Linear(4, 2)
        
# create a Sequential Class to make architetctures easier to implement

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        self.register_modules(*layers)

    def forward(self, x):
        for l in self.layers: x = l(x)
        return x

## Create an Adaptive Pool to a 1x1 and flatten it
class AdaptivePool(Module):
    def forward(self, x): return x.mean((2,3))


''' Create the CNN; pay attention to all of the components '''

def simple_cnn():
    return Sequential(
        ConvLayer(3, 16, stride=2), #32
        ConvLayer(16, 32, stride=2), #16
        ConvLayer(32, 64, stride=2), #8
        ConvLayer(64, 128, stride=2), #4
        AdaptivePool(),
        Linear(128, 10)
    )

# Add a hook 
def print_stats(outp, inp): print(outp.mean().item(), outp.std().item())
for i in range(4): m.layers[i].hook = print_stats

r = m(xbt)
print(r.shape())

''' This is a checkpoint, we now have 1) Data and 2) a model, now wer are going to create a LOSS function. '''

# negative log likelihood
def nll(input, target): return -input[range(target.shape[0]), target].mean()

# need to combine the log and the softmax 
def log_softmax(x): return (x.exp()/(x.exp().sum(-1, keepdim=True))).log()
sm = log_softmax(r)

loss = nll(sm,yb)
print(loss)

def log_softmax(x): return x - x.exp().sum(-1, keepdim=True).log()
sm = log_softmax(r)

def logsumexp(x):
     m = x.max(-1)[0]
     return m + (x-m[:, None]).exp().sum(-1).log()

# we can add the log sum to the log_softmax function 
def log_softmax(x): return x - x.logsumexp(-1, keepdim=True)

# futhermore we can  use this to create cross_entropy
def cross_entropy(preds, yb): return nll(log_softmax(preds), yb).mean()

''' Finally we must set up the learner ''' 

class SGD:
    def __init__(self, params, lr, wd=0): store_attr(self, 'params,lr,wd')
    def step(self):
        for p in self.params:
            p.data -= (p.grad.data + p.data*self.wd) * self.lr
            p.grad.data.zero_()

class DataLoaders:
    def __init__(self, *dls): self.train, self.valid = dls

dls = DataLoaders(train_dl, valid_dl)

class Learner:
    def __init__(self, model, dls, loss_func, lr, cbs, opt_func=SGD):
        store_attr(self, 'model, dls, loss_func, lr , cbs, opt_func')
        for cb in cbs: cb.learner = self 

    def one_batch(self):
        self('before_batch')
        xb, yb = self.batch
        self.preds = self.model(xb)
        self.loss = self.loss_func(self.preds, yb)
        if self.model.training:
            self.loss.backward()
            self.opt.step()
        self('after_batch')

    def one_epoch(self, train):
        self.model.training = train
        self('before_epoch')
        dl = self.dls.train if train else self.dls.valid
        for self.num, self.batch in enumerate(progress_bar(dl, leave=False)):
            self.one_batch()
        self('after_epoch')

    def fit(self, n_epochs):
        self('before_fit')
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        self.n_epochs = n_epochs
        try:
            for self.epoch in range(n_epochs):
                self.one_epoch(True)
                self.one_epoch(False)
        except CancelFitException: pass
        self('after_fit')

    def __call__(self, name):
        for cb in self.cbs: getattr(cb, name, noop)()

''' Create a callback '''
for cb in cbs:cb.learner = self

class Callback(GetAttr): _default='learner'

## move all learners to point to the gpu 
class setupLearnerCB(Callback):
    def before_batch(self):
        xb, yb = to_device(self.batch)
        self.learner.batch = tfm_x(xb), yb

    def before_fit(self): self.model.cuda()

## track and print progress
class TrackResults(Callback):
    def before_epoch(self): self.accs, self.losses, self.ns = [],[],[]

    def after_epoch(self):
        n = sum(self.ns)
        print(self.epoch, self.model.training,
            sum(self.losses).item()/n,
            sum(self.accs).item()/n)

    def after_batch(self):
        xb, yb = self.batch
        acc = (self.preds.argmax(dim=1)== yb.float().sum())
        self.accs.append(acc)
        n = len(xb)
        self.losses.append(self.loss*n)
        self.ns.append(n)


# use tghe learner 
cbs = [setupLearnerCB(), TrackResults()]
learn = Learner(simple_cnn(), dls, cross_entropy, lr= 0.1, cbs=cbs)
learn.fit(1)


''' Schedule the learning rate '''

class LRFinder(Callback):
    def before_fit(self):
        self.losses, self.lrs = [],[]
        self.learner.lr = 1e-6

    def before_batch(self):
        if not self.model.training: return
        self.opt.lr *= 1.2

    def after_batch(self):
        if not self.model.training: return
        if self.opt.lr>10 or torch.isnan(self.loss): raise CancelFitException
        self.losses.append(self.loss.item())
        self.lrs.append(self.opt.lr)

# set up and take a look at the results 
lrfind = LRFinder()
learn = Learner(simple_cnn(), dls, cross_entropy, lr=0.1, cbs=cbs+[lrfind])
learn.fit(2)

plt.plot(lrfind.lrs[:, -2], lrfind.losses[:, -2])
plt.xscale('log')

# onecycle training call back 

class OneCycle(Callback):
    def __init__(self, base_lr): self.base_lr = base_lr
    def before_fit(self): self.lrs = []

    def before_batch(self):
        if not self.model.training: return
        n = len(self.dls.train)
        bn = self.epoch*n + self.num
        mn = self.n_epochs*n
        pct = bn/mn
        pct_start, div_start = 0.25, 10
        if pct < pct_start:
            pct /= pct_start
            lr = (1-pct)*self.base_lr/div_start + pct*self.base_lr
        else:
            pct = (pct-pct_start)/(1-pct_start)
            lr = (1-pct)*self.base_lr
        self.opt.lr = lr
        self.lrs.append(lr)

# run the cycle
onecyc = OneCycle(0.1)
learn = Learner(simple_cnn(), dls, cross_entropy, lr=0.1, cbs=cbs+[onecyc])








