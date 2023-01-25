import fastbook
fastbook.setup_book()
from fastai.vision.all import *
from fastbook import *

path = untar_data(URLs.MNIST_SAMPLE)

Path.BASE_PATH = path

stacked_sevens = torch.stack([tensor(Image.open(o)) for o in (path/'train'/'7').ls()])
stacked_threes = torch.stack([tensor(Image.open(o)) for o in (path/'train'/'3').ls()])
stacked_sevens = stacked_sevens.float()/255
stacked_threes = stacked_threes.float()/255

valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = valid_7_tens.float()/255

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(stacked_threes) + [0]*len(stacked_sevens)).unsqueeze(1)

valid_x = torch.cat([valid_3_tens,valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)

dset = list(zip(train_x, train_y))
valid_dset = list(zip(valid_x,valid_y))

dl = DataLoader(dset,batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

def mnist_loss(predictions, targets):
  predictions = predictions.sigmoid()
  return torch.where(targets == 1, 1-predictions, predictions).mean()

def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()

def batch_accuracy(xb,yb):
  preds = xb.sigmoid()
  correct = (preds>0.5) == yb
  return correct.float().mean()

def validate_epoch(model):
  accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
  return round(torch.stack(accs).mean().item(), 4)

dls = DataLoaders(dl, valid_dl)

simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)

learn = Learner(dls, simple_net, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(20,0.1)