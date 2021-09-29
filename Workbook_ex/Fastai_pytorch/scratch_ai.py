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
    
