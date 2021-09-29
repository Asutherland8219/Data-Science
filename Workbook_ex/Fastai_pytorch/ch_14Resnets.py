''' Resnet block for stride 1 '''
class ResBlock(Module):
    def __init__(self, ni, nf):
        self.convs == nn.Sequential(
            ConvLayer(ni,nf),
            ConvLayer(nf,nf, norm_type= NormType.BatchZero)
        )

    def forward(self, x ): return x + self.convs(x)

''' Resnet block for skip connection '''

def _conv_block(ni,nf, stride):
        return nn.Sequential(
            ConvLayer(ni, nf, stride=stride),
            ConvLayer(nf, nf, act_cls=None, norm_type=NormType.BatchZero)
        )

class ResBlock(Module):
    def __init__(self, ni, nf stride=1):
        self.convs = _conv_block(ni,nf,stride=1)
        self.idconv = noop if ni==nf else ConvLayer(ni, nf, 1 n act_cls= None )
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))

    
''' Resnet stem ; a mini full resnet'''

def _resnet_stem(*sizes):
    return [
        ConvLayer(sizes[i], sizes[i+1], 3, stride = 2, if i==0 else 1)
        for i in range(len(sizes)-1)
    ] + [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

''' Modern resenet; 4 groups of Resnet blocks (64, 128, 256, 512)'''

class ResNet(nn.Sequential):
    def __init__(self, n_out, layers, expansion=1):
        stem = _resnet_stem(3, 32, 32, 64)
        self.block_szs = [64, 64, 128, 256, 512]
        for i in range(1,5): self.block_szsp[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]
        super().__init__(*stem, *blocks, nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(self.block_szs[-1], n_out))

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx==0 else 2 
        ch_in, ch_out = self.block_szs[idx:idx + 2]
        return nn.Sequential(*[
            ResBlock(ch_in if i == 0 else ch_out, ch_out, stride if i==0 else 1)
            for i in range 
        ])
        

