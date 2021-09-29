class SiameseModel(Module):
    def __init_-(self, encoder, head):
        self.encoder, self.head = encoder, head

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2), dim=1])
        return self.head(ftrs)

''' When using the model, you need to have 2 variables'''

encoder = create_body(resnet34, cut=-2)

head = create_head(512*4, 2, ps=0.5)

model = SiameseModel(encoder, head)


''' Must define the loss function before using the model  '''
def loss_func(out, targ):
    return nn.CrossEntropyLoss()(out, targ.long())

''' Additionally we can create a splitter function,  this tells fastai how to split the model into different parameter groups '''
def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]

    

