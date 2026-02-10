import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from models.m1_model import UpsampleCNN
from models.m2_model import ImageUpscaler
from models.m3_model import SRCNN

# PATH = '/home/wmz2007/'
PATH = './'

class SRCNN_loss():
    def __init__(self):
        super(SRCNN_loss, self).__init__()
    
    def forward(self, x, y):
        h_diff = abs(x.shape[2] - y.shape[2])
        w_diff = abs(x.shape[3] - y.shape[3])
        w_margin = w_diff // 2
        h_margin = h_diff // 2
        y_sub_img = y[:, :, w_margin:(y.shape[2] - w_margin) , h_margin:(y.shape[3] - h_margin)]
        return torch.square(y_sub_img - x).sum()
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

def train_loop(dataloader, device, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device))
        # loss = loss_fn(pred, y.to(device))
        loss = loss_fn.forward(pred, y.to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 2 == 0: 
            # Keep in mind how many batchs (number of enumerate) is tied to batch size in dataloader
            loss, current = loss.item(), batch * batch_size + X.shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, device, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            # correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")

# Example usage and training setup
if __name__ == "__main__":
    
    if sys.argv[1] == 'train':
        # Create model
        # model = ImageUpscaler(scale_factor=2, num_channels=1, num_residual_blocks=8, base_channels=64)
        # model = UpsampleCNN()
        model = SRCNN()
        
        # Get Data
        dataset = pd.read_pickle('./data/misc/data.pkl')
        input_image = dataset.loc[0, 'img_l']
        train_dataset = myDataset(dataset)
        test_dataset = myDataset(dataset, 'test')

        batch_size = 50
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Loss function (common choices for super-resolution)
        # criterion = nn.MSELoss()
        criterion = SRCNN_loss() #SRCNN Loss as in paper
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        epochs = 10
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_loader, device, model, criterion, optimizer, batch_size)
            test_loop(test_loader, device, model, criterion)
        print("Done!")
        torch.save(model.state_dict(), os.path.join(PATH, 'model1.pt'))

    elif sys.argv[1] == 'sample':
        # model = ImageUpscaler(scale_factor=2, num_channels=1, num_residual_blocks=8, base_channels=64)
        # model.load_state_dict(torch.load(os.path.join(PATH, 'model1.pt'), weights_only=True))
        model = SRCNN()
        

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(os.path.join(PATH, 'model1_colab.pt'), weights_only=True, map_location=device))
        model.to(device)

        dataset = pd.read_pickle('./data/misc/data.pkl')

        train_dataset = myDataset(dataset)
        test_dataset = myDataset(dataset, 'test')
        
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

        imgs, targets = next(iter(train_loader))
        fig, ax = plt.subplots(3)
        ax[0].imshow(imgs[0][0])
        ax[1].imshow(targets[0][0])

        output = model(imgs)[0, 0, :, :].detach().cpu()
        print(imgs.shape, output.shape)
        ax[2].imshow(output)
        plt.show()
        