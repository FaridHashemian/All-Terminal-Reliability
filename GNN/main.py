import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import math
import networkx as nx
import pandas as pd
import ast
import torch
import torch_geometric
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential,Dropout, Sigmoid
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.utils import degree
import shutil
import random
from PNA_model import PNA_Net
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from graph_dataset import GraphDataset, split_data, EarlyStopping




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1871, type=int, help="seed number")
    parser.add_argument("--path", default='/Downloads/All-Terminal-Reliability/GNN/data/', type=str, help="path to the data folder")
    parser.add_argument("--train_portion", default=0.75, type=float, help="train proportion")
    parser.add_argument("--valid_portion", default=0.15, type=float, help="validation proportion")
    parser.add_argument("--n_embed", default=128, type=int, help="node embedding size")
    parser.add_argument("--e_embed", default=128, type=int, help="edge embedding size")
    parser.add_argument("--aggs", nargs='+' , help = "choose between different aggregators: sum, mean, min, max, var and std")
    parser.add_argument("--scalers", nargs='+' , help = "choose between different scalers: identity, amplification, attenuation, linear and inverse_linear")
    parser.add_argument("--n_pna", default=4, type = int, help="number of PNA convolutions in the model")
    parser.add_argument("--hidden", default=128, type = int, help="hidden layer neuron size")
    parser.add_argument("--n_mlp_layer", default=2, type = int, help="mlp model hidden layer")
    parser.add_argument("--af", default='relu', type = str, help="choose activation function from a list of relu, elu, silu, and tanh")
    parser.add_argument("--batch_size", default=64, type = int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="weight decay for optimizer")
    parser.add_argument("--es_episodes", default=70, type = int, help="number of episodes that early stopping should wait")
    parser.add_argument("--es_improvement", default=0, type=float, help="the amount of improvement that each episode of early stopping will look for")
    parser.add_argument("--epochs", default=2, type = int, help="number of epochs")
    args = parser.parse_args()
    return args



def train_model(args):
    

    ######## Report Data



    
    ### set GNN as present working directory
    os.chdir(os.path.expanduser('~')+args.path+'/..')
    #### folder creation
    if not os.path.exists('reports'):
        os.mkdir('reports')
    
    number_of_models_ran = len(os.listdir('reports'))        
    ## models
    folder_name = 'model_num_{}'.format(number_of_models_ran + 1)
    os.makedirs('reports/'+folder_name)
    ### model configuration
    with open('reports/{}/1.model_config.txt'.format(folder_name), 'w') as f:
        f.write(str(args))
    f.close()
    ### figures
    os.makedirs('reports/{}/figures'.format(folder_name))
    #### losses
    os.makedirs('reports/{}/figures/losses'.format(folder_name))
    #### preds vs. actual
    os.makedirs('reports/{}/figures/predsvsloss'.format(folder_name))
    ### checkpoints
    os.makedirs('reports/{}/ckpnts'.format(folder_name))
    ### mse - rmse
    os.makedirs('reports/{}/errors'.format(folder_name))
    
    os.chdir(os.path.expanduser('~')+args.path)




    train_losses = []
    validation_losses = []

    ### Reading the dataset and convert it to dataloader ready for GNN
    train_data, valid_data, test_data = split_data(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    model = PNA_Net(args,train_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay= args.weight_decay)
    criterion = torch.nn.MSELoss(reduction = 'sum')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       eta_min = 0.001,
                                                       T_max = 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    es = EarlyStopping(args.es_episodes, args.es_improvement)

    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, embed = model(data.x.to(device),
                        data.edge_index.type(torch.LongTensor).to(device),
                        data.edge_attr.type(torch.LongTensor).to(device),
                        data.batch.squeeze().to(device))
            loss = criterion(out.squeeze().to(device), data.y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/len(train_loader.sampler)


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_error = 0
        for data in loader:
            data = data.to(device)
            out, embed = model(data.x.to(device),
                        data.edge_index.type(torch.LongTensor).to(device),
                        data.edge_attr.type(torch.LongTensor).to(device),
                        data.batch.squeeze().to(device))
            loss_test = criterion(out.squeeze().to(device), data.y.to(device))
            total_error += loss_test.item()
        return loss_test, total_error/len(loader.sampler)

    train_losses = []
    validation_losses = []
    best_valid = float('inf')
    for epoch in range(1, args.epochs+1):
        loss = train()
        valid_epoch_mse ,valid_mse = test(valid_loader)
        train_losses.append(loss)
        validation_losses.append(valid_mse)
        scheduler.step()
        if valid_mse < best_valid:
            torch.save(model, '../reports/{}/ckpnts/{}_ckpnts.pth'.format(folder_name, epoch))
            best_valid = valid_mse
        if es(model, valid_epoch_mse):
            #print(f'Epoch: {epoch} \t Train_Loss: {loss:.4f} \t Test_Loss: {test_mse:.4f} \t EStop:[{es.status}]')
            break
        #else:
            #print(f'Epoch: {epoch} \t Train_Loss: {loss:.4f} \t Test_Loss: {test_mse:.4f} \t EStop:[{es.status}]')

    ############################# 
    ##### Train Model Info ######
    #############################


    _epochs = [i for i in range(len(train_losses))]
    df_losses = pd.DataFrame(list(zip(_epochs, train_losses, validation_losses)),
                            columns=['epoch', 'train_loss', 'test_loss'])
    df_losses.to_csv('../reports/{}/2.losses.csv'.format(folder_name))
    
    torch.save(model, '../reports/{}/3.model.pth'.format(folder_name))

    ############################# 
    ######### Test Model ########
    #############################

    ssq = torch.nn.MSELoss(reduction = 'sum')
    running_loss_train = 0.0
    for element in train_loader:
        element.to(device)
        with torch.no_grad():
            out, embed = model(element.x.to(device),
                        element.edge_index.type(torch.LongTensor).to(device),
                element.edge_attr.type(torch.LongTensor).to(device),element.batch.squeeze().to(device))
            loss = ssq(out.squeeze(), element.y)
            running_loss_train += loss.item()
        mse = running_loss_train/len(train_loader.sampler)

    with open('../reports/{}/errors/{}_results.csv'.format(folder_name, folder_name), 'w') as f:
        f.write('train_loss\nmse, rmse\n{}, {}\n'.format(mse,mse**0.5))
    f.close()
    #print("Train Loss: %.6f \t RMSE: %.6f"%(mse,mse**0.5))

    running_loss_valid = 0.0
    for element in valid_loader:
        model.eval()
        element.to(device)
        with torch.no_grad():
            out, embed = model(element.x.to(device),
                        element.edge_index.type(torch.LongTensor).to(device),
                element.edge_attr.type(torch.LongTensor).to(device),element.batch.squeeze().to(device))
            loss = ssq(out.squeeze(), element.y)
            running_loss_valid += loss.item()
        mse = running_loss_valid/len(valid_loader.sampler)
    with open('../reports/{}/errors/{}_results.csv'.format(folder_name, folder_name), 'a') as f:
        f.write('validation_loss\nmse, rmse\n{}, {}\n'.format(mse,mse**0.5))
    f.close()
    #print("Validation Loss: %.6f \t RMSE: %.6f"%(mse,mse**0.5))

    running_loss_test = 0.0
    for element in test_loader:
        model.eval()
        element.to(device)
        with torch.no_grad():
            out, embed = model(element.x.to(device),
                        element.edge_index.type(torch.LongTensor).to(device),
                element.edge_attr.type(torch.LongTensor).to(device),element.batch.squeeze().to(device))
            loss = ssq(out.squeeze(), element.y)
            running_loss_test += loss.item()
        mse = running_loss_test/len(test_loader.sampler)
    with open('../reports/{}/errors/{}_results.csv'.format(folder_name, folder_name), 'a') as f:
        f.write('test_loss\nmse, rmse\n{}, {}\n'.format(mse,mse**0.5))
    f.close()
    #print("Test Loss: %.6f \t RMSE: %.6f"%(mse,mse**0.5))

    # Visualize learning (training loss)
    train_losses_float = [loss for loss in train_losses]
    train_loss_indices = [i for i,l in enumerate(train_losses_float)]

    valid_losses_float = [loss for loss in validation_losses]
    valid_loss_indices = [i for i,l in enumerate(valid_losses_float)]


    plot = sns.lineplot(x = train_loss_indices, y = train_losses_float, color='b', label='Train Loss')
    plot = sns.lineplot(x = valid_loss_indices , y = valid_losses_float, color= 'r', label= 'Validation Loss', linestyle=':')
    plot.legend()
    plot.set(xlabel='Epochs', ylabel='Loss', title='Losses of Train and Validation Sets')
    plot.grid('on')
    fig = plot.get_figure()
    fig.savefig("../reports/{}/figures/losses/output.png".format(folder_name))




    for dataset, name in zip([train_data, valid_data, test_data], ['Train', 'Validation', 'Test']):
        plt.clf()
        reliability_ = []
        for i in dataset:
            reliability_.append(i.y.item())

        df_ordered = pd.DataFrame(reliability_, columns=['reliability']).sort_values('reliability')
        ordered_index = list(df_ordered.index)

        data_list = []
        for i in range(len(ordered_index)):
            data_list.append(dataset[ordered_index[i]])

        #from torch_geometric.loader import DataLoader
        new_train_loader = DataLoader(data_list, batch_size=64, shuffle=False)

        prediction = []
        for element in new_train_loader:
            element.to(device)
            with torch.no_grad():
                out, embed = model(element.x.to(device),
                            element.edge_index.type(torch.LongTensor).to(device),
                        element.edge_attr.type(torch.LongTensor).to(device),element.batch.squeeze().to(device))
                for j in out:
                    prediction.append(j.item())

        
        df_ordered['prediction'] = prediction
        df_ordered.reset_index(inplace = True, drop = True)
        plt.plot(df_ordered.index, df_ordered.reliability.values, color='red', label='Actual Reliability', linewidth=2.5,)
        plt.plot(df_ordered.index, df_ordered.prediction.values, color='royalblue',linewidth=0.35,label='Predicted Reliability',alpha=0.65)
        plt.grid(True)
        plt.xlabel('Increasing Netweork Reliability Index - {} Set'.format(name))
        plt.ylabel('Reliability')
        plt.title('Prediction on {} Set'.format(name))
        plt.legend()
        plt.savefig("../reports/{}/figures/predsvsloss/{}.png".format(folder_name,name))
        #plt.show()


if __name__ == "__main__":
    args = get_args()
    train_model(args)