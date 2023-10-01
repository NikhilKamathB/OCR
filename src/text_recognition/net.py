import os
import copy
import time
import torch
import string
import random
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        '''
            Initial definition for the OCRModel class.
            Input params:
                train_loader - a torch DataLoader object representing the training data.
                val_loader - a torch DataLoader object representing the validation data.
                test_loader - a torch DataLoader object representing the testing data.
                epochs - an integer representing the number of epochs to train for. 
                device - a string representing the device to use for training.
                save_model - a boolean representing whether or not to save the model.
                save_path_dir - a string representing the path to save the model.
                saved_model - a string representing the path to the saved model.
                model_name - a string representing the name of the model.
                verbose - a boolean representing whether or not to print the training logs.
                verbose_step - an integer representing the number of steps after which to print the training logs.
                freeze_model - a boolean representing whether or not to freeze the model.
                trainable_layers - a list representing the layers to train.
                freeze_base - a boolean representing whether or not to freeze the base.
        '''
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.save_model = save_model
        self.save_path_dir = save_path_dir
        self.saved_model = saved_model
        self.model_name = model_name
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.freeze_model = freeze_model
        self.trainable_layers = trainable_layers
        self.freeze_base = freeze_base
        self.cer_metric = evaluate.load("cer")
        self.trocr_model = None
        self.train_loss = []
        self.val_loss = []
        self.start_epoch = 0
        self.skip_train = False
        self.train_step_loss = {"x": [], "y": []}
        self.save_path = self.save_path_dir + \
              self.model_name + \
                ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + '.pth' \
                    if self.save_path_dir is not None else None
        _ = self.load_model()
        if self.verbose:
            print(self.model)

    def get_model(self, model_name: str) -> None:
        '''
            This function returns a model instance.
            Input params: model_name - name of the model.
            Returns: None.
        '''
        if model_name == "TrOCRModel":
            self.trocr_model = TrOCRModel(device=self.device)
            return self.trocr_model.model
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")
    
    def load_model(self) -> object:
        '''
            This function loads a torch model.
            Input params: None
            Returns: a model object.
        '''
        self.model = self.get_model(model_name=self.model_name)
        if self.saved_model is not None:
            state_dict = torch.load(self.saved_model, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict["model_state_dict"])
            self.model.to(self.device)
            if self.verbose:
                print(f"Model loaded from {self.saved_model}.")
            self.start_epoch = state_dict["epoch"]
        else:
            print("No model loaded as `saved_model` not provided.")
        if self.freeze_model:
            if self.verbose: print("Freezing model...")
            self.freeze()
        self.best_model = copy.deepcopy(self.model)
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
    
    def compute_cer(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        '''
            This function computes the character error rate.
            Input params:
                y_hat - a tensor representing the predicted labels.
                y - a tensor representing the actual labels.
            Returns: a float representing the character error rate.
        '''
        assert self.trocr_model is not None, "TrOCR model not found."
        y_hat_string = self.trocr_model.processor.batch_decode(y_hat, skip_special_tokens=True)
        y[y == -100] = self.trocr_model.processor.tokenizer.pad_token_id
        y_string = self.trocr_model.processor.batch_decode(y, skip_special_tokens=True)
        return self.cer_metric.compute(predictions=y_hat_string, references=y_string)

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
    
    def visualize_output(self, images: torch.Tensor, y_hat: torch.Tensor, number_of_subplots: int = 8, figsize: tuple=(7, 17) ) -> None:
        '''
            This function visualizes the output of the model.
            Input params:
                images - a torch.Tensor instance.
                y_hat - a torch.Tensor instance.
                number_of_subplots - an int representing the number of subplots.
                figsize - a tuple representing the figure size.
            Returns: None.
        '''
        images_cpu = images.cpu().detach().numpy()
        y_hat_cpu_numpy = y_hat.cpu().detach().numpy()
        y_hat_string = self.trocr_model.processor.batch_decode(y_hat_cpu_numpy, skip_special_tokens=True)
        fig = plt.figure(figsize=figsize)
        for i in range(number_of_subplots):
            image = images_cpu[i, :, :, :]
            image = np.transpose(image, (1, 2, 0))
            plt.imshow(image, aspect=0.25)
            plt.title(f"Predicted String | {y_hat_string[i]}")
            plt.show()

    def train(self) -> None:
        print(f"\nDEVICE - {self.device} || EPOCHS - {self.epochs} || LEARNING RATE - {self.optimizer.param_groups[0]['lr']}.\n")
        step_size = 0
        if self.skip_train:
            print("Skipping training as model params are all frozen.")
            return (None, None)
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
                    batch_size = None
                    for key in batch:
                        if batch_size is None:
                            batch_size = batch[key].size(0)
                        batch[key] = batch[key].to(self.device)
                    step_size += batch_size
                    y_hat = self.model(**batch)
                    loss = y_hat.loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
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
                # Validating TrOCR model
                if self.trocr_model is not None:
                    with torch.no_grad():
                        for step, (val_batch) in enumerate(self.val_loader):
                            outputs = self.model.generate(val_batch["pixel_values"].to(self.device))
                            cer = self.compute_cer(y_hat=outputs, y=val_batch["labels"])
                            running_val_loss += cer
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
            return (self.model.to("cpu"), self.best_model.to("cpu"))

    def test(self) -> None:
        '''
            This function tests the model.
            Input params: None.
            Returns: None.
        '''
        print("Testing model...")
        self.best_model.eval()
        self.best_model.to(self.device)
        running_test_loss = 0
        start_epoch_test_time = time.time()
        verbose_images, verbose_y_hat = None, None
        # Testing TrOCR model
        if self.trocr_model is not None:
            with torch.no_grad():
                for step, (test_batch) in tqdm(enumerate(self.test_loader)):
                    outputs = self.best_model.generate(test_batch["pixel_values"].to(self.device))
                    cer = self.compute_cer(y_hat=outputs, y=test_batch["labels"])
                    running_test_loss += cer
                    if self.verbose and step == 0:
                        verbose_images, verbose_y_hat = test_batch["pixel_values"], outputs
                        break
        print(f'\n TEST-LOSS - {(running_test_loss/len(self.test_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_test_time):.2f}s.\n')
        if self.verbose:
            self.visualize_output(images=verbose_images, y_hat=verbose_y_hat)