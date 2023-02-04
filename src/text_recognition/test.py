import copy
from numpy import imag
import torch
import wandb
from .config import config as conf
from .utils import *


class Test:

    def __init__(self, model, test_loader, device, save_path, test_run=1, verbose=True, label_convertor=None, wandb_needed=False):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.save_path = save_path
        self.test_run = test_run
        self.verbose = verbose
        self.label_convertor = label_convertor
        self.wandb = wandb_needed
    
    def load(self):
        checkpoint = torch.load(self.save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def decode_prediction(self, logits):
        tokens = logits.softmax(2).argmax(2)
        return tokens.permute(1, 0)

    def test(self, rows=5, columns=4, config_save_model_path=None, wandb_base_path=None):
        assert rows*columns < conf.TEST_BATCH_SIZE, 'rows*columns must be less than conf.TEST_BATCH_SIZE'
        self.load()
        self.model.eval()
        for step, (images, encoded_texts, lengths) in enumerate(self.test_loader):
            if images.size(0) == conf.TEST_BATCH_SIZE:
                with torch.no_grad():
                    images, encoded_texts, lengths = images.to(self.device), encoded_texts.to(self.device), lengths.to(self.device)
                    output = self.model(images)
                    prediction, _ = output[0], output[1]
                    predicted_labels = self.decode_prediction(prediction)
                    __images__ = copy.deepcopy(images)
                if self.verbose:
                    images = images.cpu().detach()
                    predicted_labels = predicted_labels.cpu().detach().numpy() 
                    encoded_texts = encoded_texts.cpu().detach().numpy()
                    visualize(data=images, labels=predicted_labels, true_labels=encoded_texts, label_convertor=self.label_convertor, wandb_needed=self.wandb, is_ploting_prediction=True)
                if step+1 == self.test_run:
                    break
        # if self.wandb:
        #     wandb.save(self.save_path, base_path=wandb_base_path)
        #     torch.onnx.export(self.model, __images__, config_save_model_path)
        #     wandb.save(config_save_model_path, base_path=wandb_base_path)