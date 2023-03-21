"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch

from train.metrics import accuracy_TU as accuracy

def train_epoch_sparse(model, optimizer, device, data_loader, epoch, model_name):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, data in enumerate(data_loader):
        batch_graphs = data[0]
        batch_labels = data[1]

        if model_name == 'PseudoGraphormer':
            try:
                batch_spatial_baises = data[2]
                batch_spatial_baises = batch_spatial_baises.to(device)
            except:
                raise Exception('No spatial biases for model{}'.format(model_name))
    
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        try:
            if model_name == 'SAGraphTransformer':
                eigvals = batch_graphs.ndata['EigVals'].to(device)
                eigvecs = batch_graphs.ndata['EigVecs'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, eigvecs, eigvals)
            elif model_name == 'PseudoGraphormer':
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_spatial_baises)
            elif model.pe_layer.pos_enc:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            elif model.pe_layer.learned_pos_enc:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            elif model.pe_layer.n_gape > 1:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            else:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_graphs.ndata['pos_enc'])
        except:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels) 
        loss.backward()
        # nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch, model_name):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            batch_graphs = data[0]
            batch_labels = data[1]

            if model_name == 'PseudoGraphormer':
                try:
                    batch_spatial_biases = data[2]
                    batch_spatial_biases = batch_spatial_biases.to(device)
                except:
                    raise Exception('No spatial biases for model: {}'.format(model_name))

            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            try:
                if model_name == 'SAGraphTransformer':
                    eigvals = batch_graphs.ndata['EigVals'].to(device)
                    eigvecs = batch_graphs.ndata['EigVecs'].to(device)
                    batch_scores = model.forward(batch_graphs, batch_x, batch_e, eigvecs, eigvals)
                elif model_name == 'PseudoGraphormer':
                    batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_spatial_biases)
                else:
                    batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                    batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc


