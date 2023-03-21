"""
    IMPORTING LIBS
"""

import numpy as np
import os
import time
import random
import glob
import argparse, json
import pprint

import torch

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from data.positional_encs import add_automaton_encodings, add_multiple_automaton_encodings, add_random_orientation, add_rw_pos_encodings, add_spd_encodings, add_spectral_decomposition, load_encodings
from utils.main_utils import DotDict, gpu_setup, view_model_param, get_logger, add_args, setup_dirs, get_parameters, get_net_params

logger = None


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.CYCLES_graph_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset



"""
    TRAINING CODE
"""
def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, config_file, config):
    test_history = []
    val_history = []
    train_history = []

    for seed in params['seed_array']:
        logger.info(f"[!] Starting seed: {seed} in {params['seed_array']}...")
        t0 = time.time()
        per_epoch_time = []
            
        DATASET_NAME = dataset.name

        device = net_params['device']
        model = gnn_model(MODEL_NAME, net_params)
        model = model.to(device)

        if net_params['pos_enc']:
            logger.info("[!] Adding Laplacian graph positional encoding.")
            dataset._add_positional_encodings(net_params['pos_enc_dim'])
            logger.info(f'Time PE:{time.time()-t0}')

        elif net_params['adj_enc']:
            logger.info("[!] Adding adjacency matrix graph positional encoding.")
            dataset._add_adj_encodings(net_params['pos_enc_dim'])

        elif net_params.get('rand_pos_enc', False):
            logger.info(f"[!] Adding random automaton graph positional encoding ({model.pe_layer.pos_enc_dim}).")
            if net_params['diag']:
                logger.info("[!] Using diagonal weight matrix.")
            if net_params.get('n_gape', 1) > 1:
                logger.info(f"[!] Using {net_params.get('n_gape', 1)} random automata.")
                dataset = add_multiple_automaton_encodings(dataset, model.pe_layer.pos_transitions, model.pe_layer.pos_initials, net_params['diag'], net_params['matrix_type'], model)
            else:
                dataset = add_automaton_encodings(dataset, model.pe_layer.pos_transitions[0], model.pe_layer.pos_initials[0], net_params['diag'], net_params['matrix_type'], model=model)
            logger.info(f'Time PE:{time.time()-t0}')
        elif net_params.get('partial_rw_pos_enc', False):
            logger.info(f"[!] Adding partial random walk graph positional encoding ({net_params['pos_enc_dim']}).")
            if net_params['diag']:
                logger.info("[!] Using diagonal weight matrix.")
            dataset = add_rw_pos_encodings(dataset, net_params['pos_enc_dim'])
            logger.info(f'Time PE:{time.time()-t0}')

        if MODEL_NAME in ['SAGraphTransformer']:
            logger.info("[!] Adding Laplacian decompositions for spectral attention.")
            dataset = add_spectral_decomposition(dataset, net_params['pos_enc_dim'])
            logger.info(f'Time PE:{time.time()-t0}')

        if MODEL_NAME in ['PseudoGraphormer']:
            logger.info("[!] Adding shortest path distance encodings using the Floyd-Warshall algorithm.")
            dataset = add_spd_encodings(dataset)
            logger.info(f'Time PE:{time.time()-t0}')
        trainset, valset, testset = dataset.train, dataset.val, dataset.test

        if net_params.get('full_graph', False):
            st = time.time()
            logger.info("[!] Converting the given graphs to full graphs..")
            dataset._make_full_graph()
            logger.info('Time taken to convert to full graphs:',time.time()-st)    

        if net_params['num_train_data'] is not None:
            # net_params['num_train_data'] is the number of train samples to use
            from data.cycles import DGLFormDataset
            trainset = DGLFormDataset(trainset[:net_params['num_train_data']][0], trainset[:net_params['num_train_data']][1], trainset[:net_params['num_train_data']][2])

        root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs

        # Write the network and optimization hyper-parameters in folder config/
        with open(write_config_file + '.txt', 'w') as f:
            f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
            
        log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
        writer = SummaryWriter(log_dir=log_dir)

        # setting seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        
        logger.info(f"Training Graphs: {len(trainset)}", )
        logger.info(f"Validation Graphs: {len(valset)}", )
        logger.info(f"Test Graphs: {len(testset)}", )
        logger.info(f"Number of Classes: {net_params['n_classes']}", )

        optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=params['lr_reduce_factor'],
                                                        patience=params['lr_schedule_patience'],
                                                        verbose=True)
        
        epoch_train_losses, epoch_val_losses = [], []
        epoch_train_accs, epoch_val_accs = [], [] 
        

        # import train functions for all other GCNs
        from train.train_CYCLES_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

        train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
        val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
        test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

        # At any point you can hit Ctrl + C to break out of training early.
        best_test_acc = -1.0
        best_train_acc = -1.0
        try:
            for epoch in range(params['epochs']):

                logger.info(f'Epoch {epoch + 1}/{params["epochs"]}')
                start = time.time()

                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, MODEL_NAME, net_params)

                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch, MODEL_NAME)
                _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch, MODEL_NAME)

                if epoch_test_acc > best_test_acc:
                    best_test_acc = epoch_test_acc
                    best_train_acc = epoch_train_acc
                    model_dir = os.path.join(root_ckpt_dir, "MODELS_")
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    fname = f"/best_model{best_test_acc:.4f}_{params['job_num']}.pt"
                    torch.save(model.state_dict(), model_dir + fname)
                    logger.info(f"Best test accuracy: {best_test_acc:.4f} to {model_dir}")

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t = time.time() - start
                lr = optimizer.param_groups[0]['lr']
                train_loss = epoch_train_loss
                val_loss = epoch_val_loss
                train_acc = epoch_train_acc
                val_acc = epoch_val_acc
                test_acc = epoch_test_acc

                logger.info(f"""\tTime: {t:.2f}s, LR: {lr:.5f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},
                            Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}""")

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    logger.info("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    logger.info('-' * 89)
                    logger.info("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
        
        except KeyboardInterrupt:
            logger.info('-' * 89)
            logger.info('Exiting from training early because of KeyboardInterrupt')
        
        test_history.append(test_acc)
        val_history.append(val_acc)
        train_history.append(train_acc)

        _, test_acc = evaluate_network(model, device, test_loader, epoch, MODEL_NAME)
        _, train_acc = evaluate_network(model, device, train_loader, epoch, MODEL_NAME)
        logger.info("Test Accuracy: {:.4f}".format(test_acc))
        logger.info("Best Test Accuracy: {:.4f}".format(best_test_acc))
        logger.info("Train Accuracy: {:.4f}".format(train_acc))
        logger.info("Best Train Accuracy Corresponding to Best Test Accuracy: {:.4f}".format(best_train_acc))
        logger.info("Convergence Time (Epochs): {:.4f}".format(epoch))
        logger.info("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
        logger.info("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

        writer.close()

    logger.info(params)
    logger.info(f"train history: {train_history}")
    logger.info(f"test history: {test_history}")
    logger.info(f"val history: {val_history}")

    params['config'] = config_file
    params['model'] = MODEL_NAME
    params['dataset'] = DATASET_NAME

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  np.mean(np.array(test_acc))*100, np.mean(np.array(train_acc))*100, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
               


def main(args):    
    """
        USER CONTROLS
    """

    with open(args.config) as f:
        config = json.load(f)

    net_params = config['net_params']
    net_params['log_file'] = args.log_file

    global logger
    logger = get_logger(net_params['log_file'])

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'], logger)
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = get_parameters(config, args)
    # network parameters
    # net_params = config['net_params']
    net_params = get_net_params(config, args, device, params, DATASET_NAME)
    dataset = LoadData(DATASET_NAME, cycles_k=net_params['cycles_k'])

    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.num_train_data is not None:
        net_params['num_train_data'] = int(args.num_train_data)

    net_params['adj_enc'] = args.adj_enc
    net_params['dataset'] = DATASET_NAME

    dataset.train.spatial_pos_lists = []
    dataset.val.spatial_pos_lists = []
    dataset.test.spatial_pos_lists = []

    # Superpixels
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    # num_classes = len(np.unique(np.array(dataset.train[:][1])))
    net_params['n_classes'] = 2

    net_params['seed_array'] = params['seed_array']

    logger.info(pprint.pformat(net_params))
    logger.info(pprint.pformat(params))
        
    if MODEL_NAME == 'PNA':
        D = torch.cat([torch.sparse.sum(g.adjacency_matrix(transpose=True), dim=-1).to_dense() for g in
                       dataset.train.graph_lists])
        net_params['avg_d'] = dict(lin=torch.mean(D),
                                   exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1))) 

    dirs = setup_dirs(args, out_dir, MODEL_NAME, DATASET_NAME, config)
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, gnn_model, logger)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, args.config, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    main(args) 
