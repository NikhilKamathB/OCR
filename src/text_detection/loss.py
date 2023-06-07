import torch
import torch.nn as nn


class JointMSELoss(nn.Module):
    
    '''
        Custom loss function for text detection.
    '''

    def __init__(self) -> None:
        '''
            Initial definitions for the loss function.
            Returns: `None`.
        '''
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
            Input params: 
                y_hat - a tensor of shape (batch_size, 2, height, width) representing predicted region and affinity map.
                y - a tensor of shape (batch_size, 2, height, width) representing gorund truth.
            Returns: tensor representing loss.
        '''
        batch_size = y_hat.size(0)
        num_maps = y_hat.size(1)
        # print(y.max())
        # print(y_hat.max())
        maps_pred = y_hat.reshape(batch_size, num_maps, -1).split(1, dim=1)
        maps_gt = y.reshape(batch_size, num_maps, -1).split(1, dim=1)
        # print(maps_pred[0].max())
        # print(maps_pred[1].max())
        # print(maps_gt[0].max())
        # print(maps_gt[1].max())

        loss = 0
        for map in range(num_maps):
            map_pred = maps_pred[map].squeeze()
            map_gt = maps_gt[map].squeeze()
            # print(map_pred.size())
            # print(map_gt.size())
            loss += self.mse_loss(map_pred, map_gt)
        # print(loss)
        return loss