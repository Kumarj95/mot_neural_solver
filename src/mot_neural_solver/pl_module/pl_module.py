import os
import os.path as osp

import pandas as pd

from torch_geometric.data import DataLoader

import torch

from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.nn import functional as F

import pytorch_lightning as pl

from mot_neural_solver.data.mot_graph_dataset import MOTGraphDataset
# from mot_neural_solver.models.mpn2 import MOTMPNet
from mot_neural_solver.models.mpn import MOTMPNet

from mot_neural_solver.models.resnet import resnet50_fc256, load_pretrained_weights
from mot_neural_solver.path_cfg import OUTPUT_PATH
from mot_neural_solver.utils.evaluation import compute_perform_metrics
from mot_neural_solver.tracker.mpn_tracker import MPNTracker
TRACKING_OUT_COLS_CUSTOM = ['frame', 'ped_id', 'conf', 'bb_left', 'bb_top', 'bb_width', 'bb_height']

class MOTNeuralSolver(pl.LightningModule):
    """
    Pytorch Lightning wrapper around the MPN defined in model/mpn.py.
    (see https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html)

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.h1params = hparams
        self.model, self.cnn_model = self.load_model()
        self.validation_step_outputs = []     
        self.train_step_outputs=[]   
    
    def forward(self, x):
        self.model(x)

    def load_model(self):
        cnn_arch = self.h1params['graph_model_params']['cnn_params']['arch']
        model =  MOTMPNet(self.h1params['graph_model_params']).cuda()

        cnn_model = resnet50_fc256(10, loss='xent', pretrained=True).cuda()
        load_pretrained_weights(cnn_model,
                                osp.join(OUTPUT_PATH, self.h1params['graph_model_params']['cnn_params']['model_weights_path'][cnn_arch]))
        cnn_model.return_embeddings = True

        return model, cnn_model

    def _get_data(self, mode, return_data_loader = True):
        assert mode in ('train', 'val', 'test')

        dataset = MOTGraphDataset(dataset_params=self.h1params['dataset_params'],
                                  mode=mode,
                                  cnn_model=self.cnn_model,
                                  splits= self.h1params['data_splits'][mode],
                                  logger=None)

        if return_data_loader and len(dataset) > 0:
            train_dataloader = DataLoader(dataset,
                                          batch_size = self.h1params['train_params']['batch_size'],
                                          shuffle = True if mode == 'train' else False,
                                          num_workers=self.h1params['train_params']['num_workers'])
            return train_dataloader
        
        elif return_data_loader and len(dataset) == 0:
            return []
        
        else:
            return dataset

    def train_dataloader(self):
        return self._get_data(mode = 'train')

    def val_dataloader(self):
        return self._get_data('val')

    def test_dataset(self, return_data_loader=False):
        return self._get_data('test', return_data_loader = return_data_loader)

    def configure_optimizers(self):
        optim_class = getattr(optim_module, self.h1params['train_params']['optimizer']['type'])
        optimizer = optim_class(self.model.parameters(), **self.h1params['train_params']['optimizer']['args'])

        if self.h1params['train_params']['lr_scheduler']['type'] is not None:
            lr_sched_class = getattr(lr_sched_module, self.h1params['train_params']['lr_scheduler']['type'])
            lr_scheduler = lr_sched_class(optimizer, **self.h1params['train_params']['lr_scheduler']['args'])

            return [optimizer], [lr_scheduler]

        else:
            return optimizer

    def _compute_loss(self, outputs, batch):
        # Define Balancing weight
        positive_vals = batch.edge_labels.sum()

        if positive_vals:
            pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals

        else: # If there are no positives labels, avoid dividing by zero
            pos_weight = 0

        # Compute Weighted BCE:
        loss = 0
        num_steps = len(outputs['classified_edges'])
        for step in range(num_steps):
            loss += F.binary_cross_entropy_with_logits(outputs['classified_edges'][step].view(-1),
                                                            batch.edge_labels.view(-1),
                                                            pos_weight= pos_weight)
        return loss

    def _train_val_step(self, batch, batch_idx, train_val):
        device = (next(self.model.parameters())).device
        batch.to(device)

        outputs = self.model(batch)
        loss = self._compute_loss(outputs, batch)
        logs = {**compute_perform_metrics(outputs, batch), **{'loss': loss}}
        log = {key + f'/{train_val}': val for key, val in logs.items()}
        if train_val == 'train':
            return {'loss': loss, 'log': log}

        else:
            return log

    def training_step(self, batch, batch_idx):
        a=self._train_val_step(batch, batch_idx, 'train')
        self.train_step_outputs.append(a['loss'])
        
        return a

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(self._train_val_step(batch, batch_idx, 'val'))
        return self._train_val_step(batch, batch_idx, 'val')
    
    
    def on_train_epoch_end(self) -> None:
        with open("losses.txt",'a+') as f:     
            f.write(str((sum(self.train_step_outputs)/len(self.train_step_outputs)).item())+"\n")
        return super().on_train_epoch_end()
    def on_validation_epoch_end(self):
        outputs=self.validation_step_outputs
        metrics = pd.DataFrame(outputs).mean(axis=0).to_dict()
        metrics = {metric_name: torch.as_tensor(metric) for metric_name, metric in metrics.items()}
        return {'val_loss': metrics['loss/val'], 'log': metrics}

    def track_all_seqs(self, output_files_dir, dataset, use_gt = False, verbose = False, save_res=False, save_path=None):
        tracker = MPNTracker(dataset=dataset,
                             graph_model=self.model,
                             use_gt=use_gt,
                             eval_params=self.h1params['eval_params'],
                             dataset_params=self.h1params['dataset_params'])

        constraint_sr = pd.Series(dtype=float)
        for seq_name in dataset.seq_names:
            print("Tracking", seq_name)
            if verbose:
                print("Tracking sequence ", seq_name)

            os.makedirs(output_files_dir, exist_ok=True)
            res_df, constraint_sr[seq_name] = tracker.track(seq_name, output_path=osp.join(output_files_dir, seq_name + '.txt'))
            if(save_res):
                self._save_results_to_file_custom(res_df, save_path)
            if verbose:
                print("Done! \n")


        constraint_sr['OVERALL'] = constraint_sr.mean()

        return constraint_sr

    def _save_results_to_file_custom(self, seq_df, output_file_path):
        """
        Stores the tracking result to a txt file, in MOTChallenge format.
        """
        # seq_df['conf'] = 1
        seq_df['x'] = -1
        seq_df['y'] = -1
        seq_df['z'] = -1

        seq_df['bb_left'] += 1  # Indexing is 1-based in the ground truth
        seq_df['bb_top'] += 1

        final_out = seq_df[TRACKING_OUT_COLS_CUSTOM].sort_values(by=[ 'frame','ped_id'])
        x1=final_out['bb_left']
        y1=final_out['bb_top']
        x2=x1+final_out['bb_width']
        y2=y1+ final_out['bb_height']
        
        final_out['bb_width']=x2
        final_out['bb_height']=y2
        final_out=final_out.rename(columns={"frame":"fn", "ped_id":"id", 'bb_left':'x1', 'bb_top':'y1', 'bb_width':'x2', 'bb_height':'y2'})
        print(final_out)
        final_out.to_csv(output_file_path, header=False, index=False)
        final_out.to_pickle(os.path.splitext(output_file_path)[0]+".pkl")
