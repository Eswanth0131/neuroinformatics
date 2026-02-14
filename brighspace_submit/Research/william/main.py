import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import metric
from sklearn.model_selection import train_test_split

from dataset import *
from models.m1_model import UpsampleCNN
from models.m2_model import ImageUpscaler
from models.m3_model import SRCNN
from models.m4_model import SRResNet
from models.m5_model import SuperFormer
from models.m6_model import UNet3DUpsampler
from models.smp_unet import *

from torch.amp import autocast, GradScaler

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

class image_mse_loss():
    def __init__(self):
        super(image_mse_loss, self).__init__()
    
    def forward(self, x, y):
        # h_diff = abs(x.shape[2] - y.shape[2])
        # w_diff = abs(x.shape[3] - y.shape[3])
        # w_margin = w_diff // 2
        # h_margin = h_diff // 2
        # y_sub_img = y[:, :, w_margin:(y.shape[2] - w_margin) , h_margin:(y.shape[3] - h_margin)]
        return torch.square(y - x).sum()
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

def train_loop(dataloader, device, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    # scaler = GradScaler()
    for batch, (X, y) in enumerate(dataloader):
        # print(f'torch {torch.cuda.memory_allocated()}')
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        # loss = loss_fn(pred, y.to(device))
        # print(f"X; {X.shape}, y: {y.shape}")
        loss = loss_fn.forward(pred, y)

        # Debug/guard: skip update if loss is NaN or infinite
        if not torch.isfinite(loss):
            print(f"Non-finite loss detected: {loss}. pred max/min: {pred.max().item()}/{pred.min().item()}, y max/min: {y.max().item()}/{y.min().item()}")
            continue

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 2 == 0: 
            # Keep in mind how many batchs (number of enumerate) is tied to batch size in dataloader
            loss = loss.item()
            current =  batch * batch_size + X.shape[0]
            print(f"loss: {loss}  [{current:>5d}/{size}]")

def test_loop(dataloader, device, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = dataloader.dataset.__len__()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            # scaler = GradScaler()
            # with 
            pred = model(X.to(device))
            # print(f"X; {X.shape}, y: {y.shape}")
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    test_loss /= num_batches
    print(f"Avg loss per batch: {test_loss:2E} \n")
    return test_loss
    

def inference(df, device, model,):
    """
    this function auto converts data type to float 32 for gpu inference, and adjust shape of each input image to batch size 1 and input channel 1.
    """
    model.eval()
    final_df = {}

    with torch.no_grad():
        for i in tqdm(range(df.shape[0]), desc='running inference'):
            datapoint = df.iloc[i]

            x = datapoint['img_l']
            x = x[np.newaxis, :]
            x = x[np.newaxis, :]
            x = x.astype(np.float32)
            X = torch.from_numpy(x)

            # if f'{datapoint['sample_l']}' in final_df:
            #     final_df[f'{datapoint['sample_l']}'].append(model(X.to(device)))
            # else:
            #     final_df[f'{datapoint['sample_l']}'] = [model(X.to(device))]
            final_df[f'{datapoint['sample_l']}_{datapoint['layer_l']:03d}'] = model(X.to(device)).squeeze()

    return final_df


# Example usage and training setup
if __name__ == "__main__":
    if sys.argv[1] == 'train':
        # Create model
        # model = ImageUpscaler(scale_factor=2, num_channels=1, num_residual_blocks=8, base_channels=64)
        # model = UpsampleCNN()
        model = SRResNet()
        # model = SMPUNet25D()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Get Data
        dataset = pd.read_pickle('./data/misc/data.pkl')
        train_dataset = myDataset(dataset)
        test_dataset = myDataset(dataset, 'test')
        batch_size = 50
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Training setup 
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Loss function (common choices for super-resolution)
        # criterion = nn.MSELoss()
        criterion = SRCNN_loss() #SRCNN Loss as in paper
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        epochs = 1
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_loader, device, model, criterion, optimizer, batch_size)
            test_loop(test_loader, device, model, criterion)
        print("Done!")
        torch.save(model.state_dict(), os.path.join(PATH, 'model1_5d.pt'))

    elif sys.argv[1] == 'sample':
        # model = ImageUpscaler(scale_factor=2, num_channels=1, num_residual_blocks=8, base_channels=64)
        # model.load_state_dict(torch.load(os.path.join(PATH, 'model1.pt'), weights_only=True))
        model = SRCNN()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # model.load_state_dict(torch.load(os.path.join(PATH, 'model1_5d.pt'), weights_only=True, map_location=device))
        model.to(device)

        dataset = pd.read_pickle('./data/misc/data.pkl')
        train_dataset = myDataset(dataset)
        test_dataset = myDataset(dataset, 'test')
        batch_size = 50
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        imgs, targets = next(iter(train_loader))
        fig, ax = plt.subplots(4)
        ax[0].imshow(imgs[0][0]) # Input Image
        ax[1].imshow(targets[0][0]) # Target Image

        print(imgs.shape)
        imgs = imgs[:,np.newaxis].to(device)
        model_output = model(imgs)[0, 0, :, :].detach().cpu() 
        print(imgs.shape, model_output.shape) # Our Model Image
        ax[2].imshow(model_output)

        # Baseline Image
        base_ouput = F.interpolate(imgs, size=(179, 221), mode='bicubic')[0, 0, :, :].detach().cpu() 
        ax[3].imshow(base_ouput)
        plt.show()

    elif sys.argv[1] == 'super':
        model = SRResNet()
        # model = SMPUNet25D()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Create Dataset
        dataset = create_3D_Image_DF()
        train_dataset = SuperFormer_Dataset(dataset)
        test_dataset = SuperFormer_Dataset(dataset, 'test')
        batch_size = 3
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        imgs, targets = next(iter(train_loader))
        fig, ax = plt.subplots(4)
        ax[0].imshow(imgs[0][0]) # Input Image
        ax[1].imshow(targets[0][0]) # Target Image

        imgs = imgs.squeeze()
        imgs = imgs[:, np.newaxis, ...]
        imgs = imgs.to(device)

        print(f"DataLoader imgs shape:{imgs.shape}")
        # imgs = imgs[:,np.newaxis].to(device)
        imgs = F.interpolate(input=imgs, size=(179, 221, 200), mode='trilinear')
        print(f"interpolate imgs shape:{imgs.shape}")

        model_output = model(imgs)[0, 0, :, :].detach().cpu() 
        print(imgs.shape, model_output.shape) # Our Model Image
        ax[2].imshow(model_output)

        # Baseline Image
        base_ouput = F.interpolate(imgs, size=(179, 221), mode='bicubic')[0, 0, :, :].detach().cpu() 
        ax[3].imshow(base_ouput)
        plt.show()

    elif sys.argv[1] == 'final':
        z_ratio = 2

        # Create Dataset
        # dataset = create_3D_Image_DF()
        dataset = pd.read_pickle(os.path.join(PATH, 'data/misc/data3D.pkl'))

        # Split (train / val) vs (test)
        test_size = 0.9
        train_val_set, test_set = train_test_split(dataset, test_size=test_size)

        batch_size = 1
        train_val_dataset = image_3D_dataset(train_val_set)
        test_dataset = image_3D_dataset(test_set)
        train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Get Input and target image and their meta data
        torch_dataset = image_3D_dataset(dataset)
        torch_loader = DataLoader(torch_dataset)
        inputs, targets = next(iter(torch_loader))
        inputs_shape = list(inputs.shape)
        targets_shape = list(targets.shape)
        B_input, C_input, X_input, Y_input, Z_input = inputs_shape
        B_target, C_input, X_target, Y_target, Z_target = targets_shape
        print(f"DataLoader imgs shape:{inputs_shape}")
        print(f"DataLoader target shape:{targets_shape}")

        # Plotting image
        fig, ax = plt.subplots(2, 2)
        #   Input Image
        ax[0, 0].imshow(inputs[0, 0][..., Z_input // z_ratio]) 
        #   Target Image
        ax[1, 0].imshow(targets[0, 0][..., Z_target // z_ratio])

        if len(sys.argv) == 3 and sys.argv[2] == 'train':
            total_k_loss = 0
            k_split = 10
            best_model = None
            best_loss = np.inf
            for k in range(k_split):
                print('=' * 5, f"Starting Fold: {k} ", '=' * 5)
                # Load Model
                # model = SRResNet()
                # model = SMPUNetV225D()
                model = UNet3DUpsampler()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)

                # Split (Train) vs (Val)
                train_set, val_set = train_test_split(dataset, test_size=(1/k_split))
                train_set = image_3D_dataset(train_set)
                val_set = image_3D_dataset(val_set)
                train_set_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                val_set_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
            
                # Loss function
                criterion = nn.MSELoss() 

                # Optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                limit, counter = 3, 0
                for e in range(10):  # Epochs
                    print(f"Epoch {e+1}, Fold {k}\n-------------------------------")
                    train_loop(train_set_loader, device, model, criterion, optimizer, batch_size)
                    val_loss = test_loop(val_set_loader, device, model, criterion)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        counter = 0
                        torch.save(model.state_dict(), os.path.join(PATH, 'k_fold_model.pt'))
                    else:
                        counter += 1
                        if counter > limit:
                            print(f'WARNING OVERFIT, Consecutive {counter}')
                            break
                break

            print(f'Average Loss over K-{k_split}: {total_k_loss / k_split}')

            print(f'Final Test\n-------------------------------')
            val_loss = test_loop(test_loader, device, model, criterion)
            print(val_loss)

        # torch.save(best_model.state_dict(), os.path.join(PATH, 'k_fold_model.pt'))

        #   Model Image Output
        inputs = inputs.to(device)
        ouput = model.forward(inputs).detach().cpu()
        print(f'output shape: {ouput.shape}\nsum: {ouput.sum()}')
        # ax[1, 1].imshow(ouput[0, 0][..., Z_target // z_ratio])
        ax[1, 1].imshow(ouput[0, 0, ..., Z_target // z_ratio])

        #   Baseline Image
        print(f"input: {inputs[0].shape}")
        base_ouput = F.interpolate(
            inputs, 
            size=(X_target, Y_target, Z_target),
            mode='trilinear'
        )[0, 0, :, :, Z_target // z_ratio].detach().cpu()
        ax[0, 1].imshow(base_ouput)

        plt.savefig(os.path.join(PATH, './images/resnet_results.png'))
        # plt.show()

    elif sys.argv[1] == 'submit':
        # model = SRResNet()
        model = UNet3DUpsampler()
        model.load_state_dict(torch.load(os.path.join(PATH, 'k_fold_model.pt'), weights_only=True))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        submit_files = os.listdir(os.path.join(PATH, 'data/kaggle/test/low_field'))
        submit_paths = []
        for f in submit_files:
            submit_paths.append(
                os.path.join(PATH, 'data/kaggle/test/low_field', f)
            )
        dataset = whole_img_df('low', submit_paths)
        # print(dataset.head(3))
        final_submit = {}
        for i in range(dataset.shape[0]):
            final_submit[f'sample']
            print(dataset.iloc[i])


        # predictions = inference(dataset, device, model)
        # test = list(predictions.keys())
        # print(predictions[test[0]].shape, len(test), predictions[test[0]])

        # submission_df = create_submission_df(predictions)
        # print(submission_df.head(10))


