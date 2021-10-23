import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from ge import DeepWalk
import random
from utils import *
import dgl
from dataset import *
from sklearn.preprocessing import normalize

def test(model, loader, x, edge_index, device):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    model.eval()
    y_pred = []
    y_label = []

    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):
            label = label.cuda(device)
            output = model(x, edge_index, inp)
            output = m(output)

            n = torch.squeeze(output)
            loss = loss_fct(n, label.float())

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss, outputs


def train_model(model, optimizer, x, edge_index, x_target, edge_index_target,
                train_loader, val_loader, test_loader, device):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    loss_history = []
    max_auc = 0

    # Train model
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    for epoch in range(100):
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (label, inp) in enumerate(train_loader):
            label = label.cuda(device)
            # print(inp[0].shape)
            model.train()
            optimizer.zero_grad()
            output = model(x, edge_index, inp)
            output = m(output)

            n = torch.squeeze(output)
            
            loss_train = loss_fct(n, label.float())
            
            loss_history.append(loss_train)
            loss_train.backward()
            optimizer.step()
            with torch.no_grad():
                model.fc.weight[0][0].clamp_(0.0001,100)

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))

        roc_train = roc_auc_score(y_label_train, y_pred_train)

        # validation after each epoch
        roc_val, prc_val, f1_val, loss_val,outputs_val = test(model, val_loader,  x, edge_index, device = device)
        if roc_val > max_auc:
            model_max = copy.deepcopy(model)
            max_auc = roc_val
            # torch.save(model, path)
        print('epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'auroc_train: {:.4f}'.format(roc_train),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'auc_val: {:.4f}'.format(roc_val),
              'ap_val: {:.4f}'.format(prc_val),
              'f1_val: {:.4f}'.format(f1_val),
              'time: {:.4f}s'.format(time.time() - t))

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    # plt.plot(loss_history)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test,outputs = test(model_max, test_loader, x_target, edge_index_target, device=device)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auc_test: {:.4f}'.format(auroc_test),
          'ap_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))
    print(outputs)
    return np.array([auroc_test, prc_test])

def train_model_plus(model, optimizer, x, edge_index, x_target, edge_index_target, 
                id_train_positive, id_train_negative,train_matrix, features, 
                val_loader, test_loader, PE_dim, PE_method, device):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    loss_history = []
    max_auc = 0
    loss_val_history = []
    # model.cuda()
    # features = features.cuda()
    # adj = adj.cuda()

    # Train model
    t_total = time.time()
    model_max = copy.deepcopy(model)
    #np.random.seed(0)
    random.shuffle(id_train_positive)
    #random.shuffle(id_train_negative)
    slice_num = int(len(id_train_positive)/10)
    positive_train = [id_train_positive[i:i+slice_num] for i in range(0,len(id_train_positive),slice_num)]
    #negative_train = [id_train_negative[i:i+slice_num] for i in range(0,len(id_train_negative),slice_num)]
    print('Start Training...')
    
    pipe_train_x_list = []
    pipe_train_edge_index_list = []

    for j in range(10):
        print(j)
        id_train_pos = positive_train[j]
        
        pipe_train_matrix = np.copy(train_matrix)
        pipe_train_matrix[np.array(id_train_pos).T[0],np.array(id_train_pos).T[1]] = 0
        pipe_train_matrix[np.array(id_train_pos).T[1],np.array(id_train_pos).T[0]] = 0
        
        if PE_method == 'DW':
            #deepwalk
            G = nx.DiGraph(pipe_train_matrix)
            model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
            model_emb.train(window_size=5,iter=3,embed_size = PE_dim)# train model
            emb = model_emb.get_embeddings()# get embedding vectors
            embeddings = []
            for i in range(len(emb)):
                embeddings.append(emb[i])
            embeddings = np.array(embeddings)
        elif PE_method == 'LE':
        
            #LAP
            sp_adj = sp.coo_matrix(pipe_train_matrix)
            g = dgl.from_scipy(sp_adj)
            embeddings = np.array(laplacian_positional_encoding(g, PE_dim))
            embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)
        
        pipe_train_edge_index = [i for i in id_train_positive if i not in id_train_pos]
        pipe_train_x = torch.cat((torch.tensor(embeddings), features), 1)
        
    
        pipe_edge_index = np.array(pipe_train_edge_index).transpose()
        pipe_edge_index = torch.from_numpy(pipe_edge_index)
        
        pipe_train_x = pipe_train_x.unsqueeze_(0)
        pipe_edge_index = pipe_edge_index.unsqueeze_(0)

        
        pipe_train_x_list.append(pipe_train_x)
        pipe_train_edge_index_list.append(pipe_edge_index)

    
    pipe_train_x = torch.cat(pipe_train_x_list, dim=0)
    pipe_edge_index = torch.cat(pipe_train_edge_index_list, dim=0)

    
    pipe_train_x = pipe_train_x.cuda(device)
    pipe_edge_index = pipe_edge_index.cuda(device)

    auc_val_average = 0
    small_epoch_list = []
    model_list = [None, None, None, None, None, None, None, None, None, None]
    val_auc = [0] * 10
    
    for i in range(10):
        small_epoch_list.append(i)
    
    for big_epoch in range(100):
        print('-------- Big Epoch ' + str(big_epoch + 1) + ' --------')
        auc_val_average = 0
        train_loss = 0
        val_loss = 0
        random.shuffle(small_epoch_list)
        for small_epoch in small_epoch_list:
            t = time.time()
            print('-------- Small Epoch ' + str(small_epoch + 1) + ' --------')
            y_pred_train = []
            y_label_train = []
            
            random.shuffle(id_train_negative)
            negative_train = [id_train_negative[i:i+slice_num] for i in range(0,len(id_train_negative),slice_num)]
            id_train_pos = positive_train[small_epoch]
            id_train_nega = negative_train[small_epoch]
        
            id_label = []
            for i in range(len(id_train_pos)):
                id_label.append(1)
            for i in range(len(id_train_nega)):
                id_label.append(0)
            id_train = id_train_pos + id_train_nega
            train_dataset = lkpDataset(root='data', dataset='data/' + '_train',id_map=id_train, label = id_label)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
            for i, (label, inp) in enumerate(train_loader):
                label = label.cuda(device)
                # print(inp[0].shape)
                model.train()
                optimizer.zero_grad()
                output = model(pipe_train_x[small_epoch], pipe_edge_index[small_epoch],
                               inp)
                output = m(output)

                n = torch.squeeze(output)
            
                loss_train = loss_fct(n, label.float())
            
                #loss_history.append(loss_train)
                loss_train.backward()
                optimizer.step()
                with torch.no_grad():
                    model.fc.weight[0][0].clamp_(1e-5,100)

                label_ids = label.to('cpu').numpy()
                y_label_train = y_label_train + label_ids.flatten().tolist()
                y_pred_train = y_pred_train + output.flatten().tolist()

                if i % 100 == 0:
                    print('round: ' + str(small_epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                        loss_train.cpu().detach().numpy()))

            roc_train = roc_auc_score(y_label_train, y_pred_train)

            # validation after each epoch
            roc_val, prc_val, f1_val, loss_val,outputs_val = test(model, val_loader,  x, edge_index, device = device)
            val_auc[small_epoch] = roc_val
            model_list[small_epoch] = copy.deepcopy(model)
            auc_val_average += roc_val
            train_loss += loss_train.item()
            val_loss += loss_val.item()

            print('round: {:04d}'.format(small_epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(roc_train),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'auc_val: {:.4f}'.format(roc_val),
                  'ap_val: {:.4f}'.format(prc_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'time: {:.4f}s'.format(time.time() - t))
        #plt.plot(loss_history)
        auc_val_average_final = auc_val_average/10
        loss_history.append(train_loss/10)
        loss_val_history.append(val_loss/10)
        if auc_val_average_final > max_auc:
            the_num = np.where(val_auc==np.max(val_auc))[0][0]
            model_max = copy.deepcopy(model_list[the_num])
            #model_max = copy.deepcopy(model)
            max_auc = auc_val_average_final
        print('********************************************************')
        print('Epoch finished!')
        print('epoch: {:04d}'.format(big_epoch + 1),
            'auc_val: {:.4f}'.format(auc_val_average_final))
        

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
            
    #plt.plot(loss_history)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test, outputs = test(model_max, test_loader,  x_target, edge_index_target, device = device)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auc_test: {:.4f}'.format(auroc_test),
          'ap_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))
    return np.array([auroc_test, prc_test]),loss_history,loss_val_history

def train_model_ppi(model, optimizer, x_train, train_edge_index, x_val, val_edge_index,
                          x_test, test_edge_index,
                          train_loader, val_loader, test_loader, device):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    loss_history = []
    max_auc = 0

    # Train model
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    for epoch in range(100):
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (label, inp) in enumerate(train_loader):
            label = label.cuda(device)
            # print(inp[0].shape)
            model.train()
            optimizer.zero_grad()
            output = model(x_train, train_edge_index, inp)
            output = m(output)

            n = torch.squeeze(output)
            
            loss_train = loss_fct(n, label.float())
            
            loss_history.append(loss_train)
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))

        roc_train = roc_auc_score(y_label_train, y_pred_train)

        # validation after each epoch
        roc_val, prc_val, f1_val, loss_val,outputs_val = test(model, val_loader,  x_val, val_edge_index, device = device)
        if roc_val > max_auc:
            model_max = copy.deepcopy(model)
            max_auc = roc_val
            # torch.save(model, path)
        print('epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'auroc_train: {:.4f}'.format(roc_train),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'auc_val: {:.4f}'.format(roc_val),
              'ap_val: {:.4f}'.format(prc_val),
              'f1_val: {:.4f}'.format(f1_val),
              'time: {:.4f}s'.format(time.time() - t))

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    # plt.plot(loss_history)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test,outputs = test(model_max, test_loader, x_test, test_edge_index, device = device)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auc_test: {:.4f}'.format(auroc_test),
          'ap_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))
    print(outputs)
    return np.array([auroc_test, prc_test])