import copy
import time
import torch
from datetime import datetime
from datasets import load_metric
try: 
    from .models.trocr import TrOCRModel
except Exception as e:
    print("Following alternate import for net.py...")
    from models.trocr import TrOCRModel


class OCRModel:

    def __init__(self,
                 train_loader: object = None,
                 val_loader: object = None,
                 test_loader: object = None,
                 epochs: int = 10,
                 device: str = "cpu",
                 save_model: bool = True,
                 save_path_dir: str = None,
                 saved_model: str = None,
                 model_name: str = "TrOCRModel",
                 verbose = True, 
                 verbose_step = 50,
                 freeze_model: bool = False,
                 trainable_layers: list = None,
                 freeze_base: bool = False) -> None:
        pass

    def get_model(self, model_name: str) -> None:
        '''
            This function returns a model instance.
            Input params: model_name - name of the model.
            Returns: None.
        '''
        if model_name == "TrOCRModel":
            return  eval(TrOCRModel)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")
    
    def load_torch_model(self) -> object:
        '''
            This function loads a torch model.
            Input params: None
            Returns: a model object.
        '''
        self.model = self.get_model(model_name=self.model_name)
        if self.saved_model is not None:
            self.model = self.model.from_pretrained(self.saved_model)
            self.model.to(self.device)
            if self.verbose:
                print(f"Model loaded from {self.saved_model}.")
            if self.freeze_model:
                self.freeze()
            self.best_model = copy.deepcopy(self.model)
        else:
            print("No model loaded as `saved_model` not provided.")
        return self.model

    def freeze(self) -> None:
        '''
            This function freezes the model.
            Input params: None
            Returns: None.
        '''
        if self.trainable_layers is not None:
            self.skip_train = True
        for name, param in self.model.named_parameters():
            if self.trainable_layers and name in self.trainable_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if self.verbose:
            print("Model froze...")

    def configure_optimizers(self,
                             lr: int = 5e-5,
                             betas: tuple = (0.9, 0.999),
                             eps: float = 1e-8,
                             weight_decay: float = 0.01,
                             optimizer: object = None,
                             mode: str = "min",
                             patience: int = 10,
                             scheduler: object = None) -> None:
        '''
            This function configures the optimizer and scheduler.
            Input params:
                lr - a float representing the learning rate.
                betas - a tuple representing the betas for the optimizer.
                eps - a float representing the epsilon for the optimizer.
                weight_decay - a float representing the weight decay for the optimizer.
                optimizer - an optimizer object | default: AdamW.
                mode - a string representing the mode for the scheduler.
                patience - an integer representing the patience for the scheduler.
                scheduler - a scheduler object | default: ReduceLROnPlateau.
            Returns: None.
        '''
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=mode,
                patience=patience,
            )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self) -> None:
        print(f"\nDEVICE - {self.device} || EPOCHS - {self.epochs} || LEARNING RATE - {self.optimizer.param_groups[0]['lr']}.\n")
        step_size = 0
        if self.skip_train:
            print("Skipping training as model params are all frozen.")
        else:
            for epoch in range(self.start_epoch, self.epochs):
                # Train phase
                self.model.train()
                start_epoch_time = time.time()
                if self.verbose:
                    _start_at = datetime.now().strftime('%H:%M:%S %d|%m|%Y')
                    _lr = self.optimizer.param_groups[0]['lr']
                    print(f'\nEPOCH - {epoch+1}/{self.epochs} || START AT - {_start_at} || LEARNING RATE - {_lr}\n')
                running_loss, step_running_loss = 0, 0
                start_step_time = time.time()
                for step, (batch) in enumerate(self.train_loader):
                    step_size += images.size(0)
                    images, y_region_maps, y_affinity_maps = images.to(self.device), y_region_maps.to(self.device) ,y_affinity_maps.to(self.device)
                    self.optimizer.zero_grad()
                    y_hat, _ = self.model(images)
                    y = torch.permute(
                            torch.cat((torch.unsqueeze(y_region_maps, dim=1), 
                                        torch.unsqueeze(y_affinity_maps, dim=1)), 
                            dim=1),
                        dims=(0, 2, 3, 1))
                    loss = self.criterion(y_hat, y)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    step_running_loss += loss.item()
                    if self.verbose:
                        if (step+1) % self.verbose_step == 0:
                            print(
                                    f'\tTrain Step - {step+1}/{len(self.train_loader)} | ' + \
                                    f'Train Step Loss: {(step_running_loss/self.verbose_step):.5f} | ' + \
                                    f'Time: {(time.time() - start_step_time):.2f}s.\n'
                                )
                            self.train_step_loss["x"].append(step_size/1000)
                            self.train_step_loss["y"].append(step_running_loss/self.verbose_step)
                            step_running_loss = 0   
                            start_step_time = time.time()
                self.train_loss.append(running_loss/len(self.train_loader))
                self.scheduler.step(running_loss/len(self.train_loader))
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || TRAIN-LOSS - {(running_loss/len(self.train_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_time):.2f}s.\n')
                # Validation Phase
                start_epoch_val_time = time.time()
                self.model.eval()
                running_val_loss = 0
                for step, (val_images, val_y_region_maps, val_y_affinity_maps) in enumerate(self.val_loader):
                    val_images, val_y_region_maps, val_y_affinity_maps = val_images.to(self.device), val_y_region_maps.to(self.device), val_y_affinity_maps.to(self.device)
                    val_y_hat, _ = self.model(val_images)
                    val_y = torch.permute(
                                torch.cat((torch.unsqueeze(val_y_region_maps, dim=1), 
                                            torch.unsqueeze(val_y_affinity_maps, dim=1)), 
                                dim=1), 
                            dims=(0, 2, 3, 1))
                    val_loss = self.criterion(val_y_hat, val_y)
                    running_val_loss += val_loss.item()
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || VAL-LOSS - {(running_val_loss/len(self.val_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_val_time):.2f}s.\n')
                self.val_loss.append(running_val_loss/len(self.val_loader))
                if self.best_test_loss > running_val_loss:
                    self.best_test_loss = running_val_loss
                    self.best_model = copy.deepcopy(self.model)
            if self.verbose:
                self.plot_loss_curve()
                self.plot_step_loss_curve()
            if self.save_model:
                self.save(model=self.best_model)
        return self.model.to("cpu"), self.best_model.to("cpu")

    def test(self) -> None:
        pass

    def metric(self) -> object:
        pass