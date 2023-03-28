"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch

from train.metrics import MAE
from ogb.lsc import PCQM4Mv2Evaluator

def train_epoch_sparse(model, optimizer, device, data_loader, epoch, model_name, net_params):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0

    for iter, data in enumerate(data_loader):

        batch_graphs = data[0].to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = data[1].to(device)

        if model_name == 'PseudoGraphormer':
            try:
                batch_spatial_biases = data[2]
                batch_spatial_biases = batch_spatial_biases.to(device)
            except:
                raise Exception('No spatial biases for model: {}'.format(model_name))

        optimizer.zero_grad()
        try:
            if model_name == 'SAGraphTransformer':
                    eigvals = batch_graphs.ndata['EigVals'].to(device)
                    eigvecs = batch_graphs.ndata['EigVecs'].to(device)
                    batch_scores = model.forward(batch_graphs, batch_x, eigvecs, eigvals)
            elif model_name == 'PseudoGraphormer':
                batch_scores = model.forward(batch_graphs, batch_x, batch_spatial_biases)
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
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch, model_name):
    evaluator = PCQM4Mv2Evaluator()
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0

    y_pred = []
    y_true = []

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            batch_graphs = data[0].to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = data[1].to(device)

            if model_name == 'PseudoGraphormer':
                try:
                    batch_spatial_biases = data[2]
                    batch_spatial_biases = batch_spatial_biases.to(device)
                except:
                    raise Exception('No spatial biases for model: {}'.format(model_name))

            try:
                if model_name == 'SAGraphTransformer':
                    eigvals = batch_graphs.ndata['EigVals'].to(device)
                    eigvecs = batch_graphs.ndata['EigVecs'].to(device)
                    batch_scores = model.forward(batch_graphs, batch_x, eigvecs, eigvals)
                elif model_name == 'PseudoGraphormer':
                    batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_spatial_biases)
                else:
                    batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                    batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            
            y_pred.extend(batch_scores.detach().cpu().view(-1))
            y_true.extend(batch_targets.detach().cpu().view(-1))

            loss = model.loss(batch_scores, batch_targets)

            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)

        y_pred = torch.Tensor(y_pred).view(-1)
        y_true = torch.Tensor(y_true).view(-1)

        input_dict = {'y_pred': y_pred, 'y_true': y_true}
        result_dict = evaluator.eval(input_dict)

        mae = torch.tensor(result_dict['mae'])

        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)

        epoch_test_mae = mae
        
    return epoch_test_loss, epoch_test_mae



