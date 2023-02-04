import os
import time
import wandb
import torch
import string
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from .utils import *


class Train:

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device, save_path_dir, need_grap_clip=True, clip_norm=5, label_convertor=None, verbose=True, verbose_step=50, save_at=5, need_val=True, model_name="N-A", wandb_needed=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.save_path_dir = save_path_dir
        self.need_grad_clip = need_grap_clip
        self.clip_norm = clip_norm
        self.label_convertor = label_convertor
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.save_at = save_at
        self.model_name = model_name
        self.need_val = need_val
        self.start_epoch = 0
        self.train_step_loss = {"x": [], "y": []}
        self.train_step_accuracy = {"x": [], "y": []}
        self.train_loss, self.val_loss, self.train_accuracy, self.val_accuracy = [], [], [], []
        self.wandb = wandb_needed
        self.save_path = None
    
    def save(self, append_name=None, epoch=None):
        model_name = self.model_name + '_' + f'{str(int(time.time()))}' + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + '.pth'
        save_path = append_name + '_' + model_name if append_name else model_name
        self.save_path = self.save_path_dir + save_path
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step_loss': self.train_step_loss,
            'train_step_accuracy': self.train_step_accuracy,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy
            }, self.save_path)
    
    def plot_step_curve(self, x_label='Step loss'):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Training learning curves')
        ax1.plot(self.train_step_loss["x"], self.train_step_loss["y"], 'r', label='Training step loss')
        ax1.set_title('Training Step Loss')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Step Loss')
        ax1.legend()
        ax2.plot(self.train_step_accuracy["x"], self.train_step_accuracy["y"], 'r', label='Training step acc')
        ax2.set_title('Training Step Accuracy')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Step Accuracy')
        ax2.legend()
        fig.tight_layout(pad=2.0)
        if self.wandb:
            wandb.log({"Train Step Curves - Matplotlib": fig})
        plt.show()
    
    def plot_curve(self, x_label='Epochs'):
        epochs = range(1, len(self.train_loss)+1)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Learning curves')
        ax1.plot(epochs, self.train_loss, 'r', label='Training loss')
        # plt.plot(epochs, self.train_loss, 'r*', label='Training loss spots')
        if self.need_val:
            ax1.plot(epochs, self.val_loss, 'g', label='Validation loss')
            # plt.plot(epochs, self.val_loss, 'g*', label='Validation loss spots')
        ax1.set_title('Training/Validation Loss')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(epochs, self.train_accuracy, 'r', label='Training Acc')
        # plt.plot(epochs, self.train_loss, 'r*', label='Training loss spots')
        if self.need_val:
            ax2.plot(epochs, self.val_accuracy, 'g', label='Validation Acc')
            # plt.plot(epochs, self.val_loss, 'g*', label='Validation loss spots')
        ax2.set_title('Training/Validation Accuracy')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        fig.tight_layout(pad=2.0)
        if self.wandb:
            wandb.log({"Train/Validation Leaarning Curves - Matplotlib": fig})
        plt.show()
    
    def load(self, path=None):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_step_loss = checkpoint['train_step_loss']
        self.train_step_accuracy = checkpoint['train_step_accuracy']
        self.train_loss = checkpoint['train_loss']
        self.val_loss = checkpoint['val_loss']
        self.train_accuracy = checkpoint['train_accuracy']
        self.val_accuracy = checkpoint['val_accuracy']
    
    def compute_batch_accuracy(self, pred_texts, true_texts):
        correct_counts = 0
        batch_size = pred_texts.size(1)
        tokens = pred_texts.softmax(2).argmax(2)
        tokens = tokens.permute(1, 0)
        for i in range(batch_size):
            pred_texts_encodings = tokens[i].tolist()
            true_texts_encodings = true_texts[i].tolist()
            pred_string = self.label_convertor.decode(pred_texts_encodings)
            true_string = self.label_convertor.decode(true_texts_encodings)
            pred_string_process = pred_string.replace(' ', '')
            true_string_process = true_string.replace(' ', '')
            if pred_string_process == true_string_process:
                correct_counts += 1
        return correct_counts/batch_size

    def train(self):
        if self.wandb:
            wandb.watch(models=self.model, criterion=self.criterion, log="all", log_freq=10)
        print(f"\nDEVICE - {self.device} || EPOCHS - {self.epochs} || LEARNING RATE - {self.optimizer.param_groups[0]['lr']}.\n")
        metrics = {}
        step_size = 0
        for epoch in range(self.start_epoch, self.epochs):
            start_epoch_time = time.time()
            if self.verbose:
                _start_at = datetime.now().strftime('%H:%M:%S %d|%m|%Y')
                _lr = self.optimizer.param_groups[0]['lr']
                print(f'\nEPOCH - {epoch+1}/{self.epochs} || START AT - {_start_at} || LEARNING RATE - {_lr}\n')
            self.model.train()
            running_loss, running_accuracy, step_running_loss, step_running_accuracy = 0, 0, 0, 0
            start_step_time = time.time()
            for step, (images, encoded_texts, lengths) in enumerate(self.train_loader):
                step_size += images.size(0) 
                images, encoded_texts, lengths = images.to(self.device), encoded_texts.to(self.device), lengths.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                prediction, prediction_logits = output[0], output[1]
                logits_lens = torch.full(size=(prediction.size(1) ,), fill_value=prediction.size(0), dtype=torch.int32).to(self.device)
                loss = self.criterion(prediction_logits, encoded_texts, logits_lens, lengths.squeeze(1))
                accuracy = self.compute_batch_accuracy(prediction, encoded_texts)
                loss.backward()
                if self.need_grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()
                running_loss += loss.item()
                step_running_loss += loss.item()
                running_accuracy += accuracy
                step_running_accuracy += accuracy
                if self.verbose:
                    if (step+1) % self.verbose_step == 0 or (step+1) == len(self.train_loader):
                        print(
                                f'\t  -- Train Step - {step+1}/{len(self.train_loader)} | ' + \
                                f'Train Step Loss: {(step_running_loss/self.verbose_step):.5f} | ' + \
                                f'Train Step Accuracy: {(step_running_accuracy/self.verbose_step):.5f} | ' + \
                                f'Time: {(time.time() - start_step_time):.2f}s.\n'
                            )
                        self.train_step_loss["x"].append(step_size)
                        self.train_step_loss["y"].append(step_running_loss/self.verbose_step)
                        self.train_step_accuracy["x"].append(step_size)
                        self.train_step_accuracy["y"].append(step_running_accuracy/self.verbose_step)
                        step_running_loss = 0   
                        step_running_accuracy = 0
                        start_step_time = time.time()
            self.train_loss.append(running_loss/len(self.train_loader))
            self.train_accuracy.append(running_accuracy/len(self.train_loader))
            self.scheduler.step(running_loss/len(self.train_loader))
            metrics["Training Loss"] = running_loss/len(self.train_loader)
            metrics["Training Accuracy"] = running_accuracy/len(self.train_loader)
            if self.verbose:
                print(f'\tEPOCH - {epoch+1}/{self.epochs} || TRAIN LOSS - {(running_loss/len(self.train_loader)):.5f} || TRAIN ACCURACY - {(running_accuracy/len(self.train_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_time):.2f}s.\n')
            if self.need_val:
                self.model.eval()
                start_epoch_val_time = time.time()
                running_val_loss, running_val_accuracy, step_running_val_loss, step_running_val_accuracy = 0, 0, 0, 0
                start_step_val_time = time.time()
                for step, (images, encoded_texts, lengths) in enumerate(self.val_loader):
                    with torch.no_grad():
                        images, encoded_texts = images.to(self.device).float(), encoded_texts.to(self.device).float()
                        output = self.model(images)
                        prediction, prediction_logits = output[0], output[1]
                        logits_lens = torch.full(size=(prediction.size(1) ,), fill_value=prediction.size(0), dtype=torch.int32).to(self.device)
                        loss = self.criterion(prediction_logits, encoded_texts, logits_lens, lengths)
                        accuracy = self.compute_batch_accuracy(prediction, encoded_texts)
                        running_val_loss += loss.item()
                        step_running_val_loss += loss.item() 
                        running_val_accuracy += accuracy
                        step_running_val_accuracy += accuracy
                        if self.verbose:
                            if (step+1) % self.verbose_step == 0 or (step+1) == len(self.val_loader):
                                print(
                                    f'\t  -- Validation Step - {step+1}/{len(self.val_loader)} | ' + \
                                    f'Val Step Loss: {(step_running_val_loss/self.verbose_step):.5f} | ' + \
                                    f'Val Step Accuracy: {(step_running_val_accuracy/self.verbose_step):.5f} | ' + \
                                    f'Time: {(time.time() - start_step_val_time):.2f}.\n'
                                )
                                step_running_val_loss = 0 
                                step_running_val_accuracy = 0
                                start_step_val_time = time.time()
                self.val_loss.append(running_val_loss/len(self.val_loader))
                self.val_accuracy.append(running_val_accuracy/len(self.val_loader))
                metrics["Validation Loss"] = running_val_loss/len(self.val_loader)
                metrics["Validation Accuracy"] = running_val_accuracy/len(self.val_loader)
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || VALIDATION LOSS - {(running_val_loss/len(self.val_loader)):.5f} || VALIDATION Accuracy - {(running_val_accuracy/len(self.val_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_val_time):.2f}\n')
                self.scheduler.step(running_val_loss/len(self.val_loader))
            if self.wandb:
                wandb.log(metrics, step=epoch+1)
            if self.epochs == epoch+1:
                self.save(epoch=epoch+1)
            elif (epoch+1) % self.save_at == 0:
                self.save(append_name=f"EPOCH_{str(epoch+1)}", epoch=epoch)
        if self.train_step_loss['x']:
            self.plot_step_curve()
        self.plot_curve()
        return self.train_loss, self.save_path