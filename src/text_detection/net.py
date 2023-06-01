import torch
import pytorch_lightning as pl
from .models.craft import CRAFT


class LitOCRModel(pl.LightningModule):

    '''
        This class implements the OCR model using pytorch lightning.
    '''

    def __init__(self,
                 criteria: object,
                 loaders: dict,
                 learning_rate: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.01,
                 batch_size: int = 32,
                 verbose: bool = True,
                 freeze: bool = False,
                 model_name: str = "craft") -> None:
        '''
            Initial definition for the LitOCRModel class.
            Input params:
                criteria - object representing the loss function.
                loaders - a dict containing the train, validation and test data loaders.
                learning_rate - a float representing the learning rate.
                betas - a tuple representing the betas for the optimizer.
                epsilon - a float representing the epsilons for optimizer.
                weight_decay - a float representing the weight decay for the optimizer.
                batch_size - an integer representing the batch size.
                verbose - a boolean representing whether to print the printables.
                freeze - a boolean representing whether to freeze the model.
                model_name - name of the model.
            Returns: None.
        '''
        super().__init__()
        self.criteria = criteria
        self.loaders = loaders
        self.learning_rate = learning_rate
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.verbose = verbose
        self.freeze = freeze
        self.model = self.get_model(model_name=model_name)
        if self.verbose:
            print("Model Summary:\n", self.model)
    
    def get_model(self, model_name: str) -> object:
        '''
            This function returns a model instance.
            Input params: `model_name` - name of the model.
            Returns: `model` - an nn instance.
        '''
        if model_name == "craft":
            model = eval(model_name.upper())
            return model(freeze=self.freeze)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

    def forward(self, x: torch.Tensor) -> tuple:
        '''
            Input params: `x` - a torch.Tensor instance.
            Returns: a torch.Tensor instance.
        '''
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        '''
            Returns: an optimizer instance.
        '''
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.epsilon,
            weight_decay=self.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CyclicLR(
                optimizer=optimizer,
                base_lr=self.learning_rate,
                max_lr=1e-2,
                cycle_momentum=False, # False for AdamW
            ),
            "interval": "step"
        }
        return [optimizer], [scheduler]

    def train_dataloader(self) -> object:
        '''
            Returns: a train dataLoader instance.
        '''
        return self.loaders["train"]

    def val_dataloader(self) -> object:
        '''
            Returns: a validation dataLoader instance.
        '''
        return self.loaders["validation"]

    def test_dataloader(self) -> object:
        '''
            Returns: a test dataLoader instance.
        '''
        return self.loaders["test"]
    
    def execute(self, batch: tuple, log_msg: str, prog_bar: bool = True, on_step: bool = True, on_epoch: bool = True):
        '''
            This function executes the model and logs the loss.
            Input params:
                batch - a tuple representing a batch of data.
                log_msg - a str representing the log message.
                prog_bar - a boolean representing whether to show the progress bar.
                on_step - a boolean representing whether to show the progress bar on step.
                on_epoch - a boolean representing whether to show the progress bar on epoch.
            Returns: a torch.Tensor instance.
        '''
        image, y_region_map, y_affinity_map = batch
        y_hat, _ = self.forward(x=image)
        y = torch.permute(
                torch.cat((torch.unsqueeze(y_region_map, dim=1), 
                            torch.unsqueeze(y_affinity_map, dim=1)), 
                dim=1), 
            dims=(0, 2, 3, 1))
        loss = self.criteria(y_hat, y)
        self.log(log_msg, loss, prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        '''
            Input params:
                batch - a tuple representing a batch of data.
                batch_idx - an int representing the batch index.
            Returns: a torch.Tensor instance.
        '''
        return self.execute(batch=batch, log_msg="train_loss", on_step=False)

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        '''
            Input params:
                batch - a tuple representing a batch of data.
                batch_idx - an int representing the batch index.
            Returns: a torch.Tensor instance.
        '''
        return self.execute(batch=batch, log_msg="val_loss")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        '''
            Input params:
                batch - a tuple representing a batch of data.
                batch_idx - an int representing the batch index.
            Returns: a torch.Tensor instance.
        '''
        return self.execute(batch=batch, log_msg="test_loss")

    def get_progress_bar_dict(self) -> dict:
        '''
            Returns: a dict containing the progress bar metrics.
        '''
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items