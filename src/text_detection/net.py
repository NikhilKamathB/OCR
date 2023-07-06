import os
import time
import copy
import torch
import string
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
from models.craft import CRAFT


class OCRModel:

    '''
        OCRModel class is used to train/test the model and save the best model.
    '''

    def __init__(self, 
                 train_loader: object = None, 
                 val_loader: object = None,
                 test_loader: object = None,
                 criterion: object = None,
                 epochs: int = 50,
                 device: str = "cpu",
                 save_model: bool = True,
                 save_path_dir: str = None,
                 saved_model: str = None,
                 model_name: str = "craft",
                 raw_load: bool = True,
                 verbose = True, 
                 verbose_step = 50,
                 freeze_model: bool = False,
                 trainable_layers: list = None,
                 freeze_base: bool = False) -> None:
        '''
            Initial definition for the OCRModel class.
            Input params:
                train_loader - a torch dataloader instance for training data.
                val_loader - a torch dataloader instance for validation data.
                test_loader - a torch dataloader instance for test data.
                criterion - a loss function.
                epochs - an integer representing the number of epochs.
                device - a string representing the device to be used.
                save_model - a boolean representing whether to save the model.
                save_path_dir - a string representing the path to save the model.
                saved_model - a string representing the path to the saved model.
                model_name - name of the model | default: craft, available: craft.
                raw_load - a boolean representing whether to load the actual craft model.
                verbose - a boolean representing whether to print the printables.
                verbose_step - an integer representing the number of steps to print the printables.
                freeze_model - a boolean representing whether to freeze this model.
                trainable_layers - a list representing this model's layers to be trained.
                freeze_base - a boolean representing whether to freeze the base model.
            Returns: None.
        '''
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.save_model = save_model
        self.save_path_dir = save_path_dir
        self.saved_model = saved_model
        self.model_name = model_name
        self.raw_load = raw_load
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.freeze_model = freeze_model
        self.trainable_layers = trainable_layers
        self.freeze_base = freeze_base
        self.start_epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.skip_train = False
        self.train_step_loss = {"x": [], "y": []}
        self.best_test_loss = np.inf
        self.save_path = self.save_path_dir + \
              self.model_name + \
                ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + '.pth' \
                    if self.save_path_dir is not None else None
        _ = self.load_torch_model()

    def get_model(self, model_name: str) -> None:
        '''
            This function returns a model instance.
            Input params: model_name - name of the model.
            Returns: None.
        '''
        if model_name == "craft":
            model = eval(model_name.upper())
            return model(freeze=self.freeze_base)
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
            state_dict = torch.load(self.saved_model, map_location=torch.device("cpu"))
            if self.raw_load:
                start_idx = 0
                new_state_dict = OrderedDict()
                if list(state_dict.keys())[0].startswith("module"):
                    start_idx = 1
                for k, v in state_dict.items():
                    name = ".".join(k.split(".")[start_idx: ])
                    new_state_dict[name] = v
                state_dict = new_state_dict
                self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict["model_state_dict"])
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
                             lr: int = 1e-3,
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

    def save(self, model: object) -> None:
        '''
            This function saves the model.
            Input params: model - a model object.
            Returns: None.
        '''
        os.makedirs(self.save_path_dir, exist_ok=True)
        model.to("cpu")
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.save_path)
        print(f"Model saved to -> {self.save_path} | device -> cpu")
        model.to(self.device)
    
    def plot_loss_curve(self, x_label: str = 'Epochs'):
        '''
            This function plots the loss curve.
            Input params: x_label - a string representing the x label.
            Returns: None.
        '''
        epochs = range(1, len(self.train_loss)+1)
        plt.plot(epochs, self.train_loss, 'g', label='Training loss')
        plt.plot(epochs, self.train_loss, 'g*', label='Training loss spots')
        plt.plot(epochs, self.val_loss, 'r', label='Test loss')
        plt.plot(epochs, self.val_loss, 'r*', label='Test loss spots')
        plt.title('Training and testing Loss')
        plt.xlabel(x_label)
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def plot_step_loss_curve(self, x_label: str = 'Step Loss x1000') -> None:
        '''
            This function plots the step loss curve.
            Input params: x_label - a string representing the x label.
            Returns: None.
        '''
        if self.train_step_loss["x"] and self.train_step_loss["y"]:
            plt.plot(self.train_step_loss["x"], self.train_step_loss["y"], 'b', label='Training step loss')
            plt.title('Training Step Loss')
            plt.xlabel(x_label)
            plt.ylabel('Step Loss')
            plt.legend()
            plt.show()
    
    def visualize_output(self, images: torch.Tensor, y_hat: torch.Tensor, number_of_subplots: int = 15, columns: int = 3, rows: int = 5, figsize=(15, 15)) -> None:
        '''
            This function visualizes the output of the model.
            Input params:
                images - a torch.Tensor instance.
                y_hat - a torch.Tensor instance.
                number_of_subplots - an int representing the number of subplots.
                columns - an int representing the number of columns.
                rows - an int representing the number of rows.
                figsize - a tuple representing the figure size.
            Returns: None.
        '''
        assert number_of_subplots == columns * rows, "`number_of_subplots` must be equal to the product `columns` and `rows` for plotting convenience."
        images_cpu = images.cpu().detach().numpy()
        y_hat_cpu_numpy = y_hat.cpu().detach().numpy()
        _, ax = plt.subplots(rows, columns, figsize=figsize)
        image_idx = 0
        for r in range(rows):
            for c in range(0, columns, 3):
                image = images_cpu[image_idx, :, :, :]
                image = np.transpose(image, (1, 2, 0))
                region_image = y_hat_cpu_numpy[image_idx, :, :, 0]
                affinity_image = y_hat_cpu_numpy[image_idx, :, :, 1]
                ax[r, c].imshow(image)
                ax[r, c].set_title("Image")
                ax[r, c+1].imshow(region_image, cmap="gray")
                ax[r, c+1].set_title("Region Map")
                ax[r, c+2].imshow(affinity_image, cmap="gray")
                ax[r, c+2].set_title("Affinity Map")
                image_idx += 1
    
    def train(self):
        '''
            This function trains and validates the model.
            Input params: None.
            Returns: None.
        '''
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
                for step, (images, y_region_maps, y_affinity_maps) in enumerate(self.train_loader):
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
    
    def test(self):
        '''
            This function tests the model.
            Input params: None.
            Returns: None.
        '''
        print("Testing model...")
        self.best_model.eval()
        self.best_model.to("cpu")
        running_test_loss = 0
        start_epoch_test_time = time.time()
        verbose_images, verbose_y_hat = None, None
        for step, (test_images, test_y_region_maps, test_y_affinity_maps) in tqdm(enumerate(self.test_loader)):
            test_images, test_y_region_maps, test_y_affinity_maps = test_images.to("cpu"), test_y_region_maps.to("cpu"), test_y_affinity_maps.to("cpu")
            test_y_hat, _ = self.best_model(test_images)
            test_y = torch.permute(
                        torch.cat((torch.unsqueeze(test_y_region_maps, dim=1), 
                                    torch.unsqueeze(test_y_affinity_maps, dim=1)), 
                        dim=1), 
                    dims=(0, 2, 3, 1))
            test_loss = self.criterion(test_y_hat, test_y)
            running_test_loss += test_loss.item()
            if self.verbose and step == 0:
                verbose_images, verbose_y_hat = test_images, test_y_hat
        print(f'\n TEST-LOSS - {(running_test_loss/len(self.test_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_test_time):.2f}s.\n')
        if self.verbose:
            self.visualize_output(images=verbose_images, y_hat=verbose_y_hat)