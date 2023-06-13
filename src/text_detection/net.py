import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from collections import OrderedDict
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
                 saved_model: str = None,
                 raw_load: bool = True,
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
                saved_model - a string representing the path to the saved model.
                raw_load - a boolean representing whether to load the actual (original) model.
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
        self.saved_model = saved_model
        self.model = self.get_model(model_name=model_name)
        self.load_torch_model(raw_load=raw_load)
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
    
    def load_torch_model(self, raw_load: bool = True) -> None:
        '''
            This function loads a torch model.
            Input params:
                raw_load - a boolean representing whether to load the actual (original) model.
            Returns: None.
        '''
        if self.saved_model is not None:
            state_dict = torch.load(self.saved_model, map_location=torch.device("cpu"))
            if raw_load:
                start_idx = 0
                new_state_dict = OrderedDict()
                if list(state_dict.keys())[0].startswith("module"):
                    start_idx = 1
                for k, v in state_dict.items():
                    name = ".".join(k.split(".")[start_idx: ])
                    new_state_dict[name] = v
                state_dict = new_state_dict
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            if self.verbose:
                print(f"Model loaded from {self.saved_model}.")
        else:
            print("No model loaded as `saved_model` not provided.")

    def forward(self, x: torch.Tensor) -> tuple:
        '''
            Input params: `x` - a torch.Tensor instance.
            Returns: a torch.Tensor instance.
        '''
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        '''
            Input params: None.
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
            Input params: None.
            Returns: a train dataLoader instance.
        '''
        return self.loaders["train"]

    def val_dataloader(self) -> object:
        '''
            Input params: None.
            Returns: a validation dataLoader instance.
        '''
        return self.loaders["validation"]

    def test_dataloader(self) -> object:
        '''
            Input params: None.
            Returns: a test dataLoader instance.
        '''
        return self.loaders["test"]
    
    def execute(self, batch: tuple, log_msg: str, prog_bar: bool = True, on_step: bool = True, on_epoch: bool = True, mode: str = "train"):
        '''
            This function executes the model and logs the loss.
            Input params:
                batch - a tuple representing a batch of data.
                log_msg - a str representing the log message.
                prog_bar - a boolean representing whether to show the progress bar.
                on_step - a boolean representing whether to show the progress bar on step.
                on_epoch - a boolean representing whether to show the progress bar on epoch.
                mode - a str representing the mode.
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
        return self.execute(batch=batch, log_msg="val_loss", mode="validation")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        '''
            Input params:
                batch - a tuple representing a batch of data.
                batch_idx - an int representing the batch index.
            Returns: a torch.Tensor instance.
        '''
        return self.execute(batch=batch, log_msg="test_loss", mode="test")

    def on_test_epoch_end(self) -> None:
        '''
            This function is called at the end of the test epoch.
            Here we visualize the output of the model.
            Input params: None.
            Returns: None.
        '''
        image, _, _ = next(iter(self.test_dataloader()))
        image = image.to(self.device)
        y_hat, _ = self.forward(x=image)
        if self.verbose:
            self.visualize_output(y_hat=y_hat)

    def get_progress_bar_dict(self) -> dict:
        '''
            Returns: a dict containing the progress bar metrics.
        '''
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def visualize_output(self, y_hat: torch.Tensor, number_of_subplots: int = 8, columns: int = 4, rows: int = 2, figsize=(30, 10)) -> None:
        '''
            This function visualizes the output of the model.
            Input params:
                y_hat - a torch.Tensor instance.
                number_of_subplots - an int representing the number of subplots.
                columns - an int representing the number of columns.
                rows - an int representing the number of rows.
                figsize - a tuple representing the figure size.
            Returns: None.
        '''
        assert number_of_subplots == columns * rows, "`number_of_subplots` must be equal to the product `columns` and `rows` for plotting convenience."
        y_hat_cpu_numpy = y_hat.cpu().detach().numpy()
        _, ax = plt.subplots(rows, columns, figsize=figsize)
        image_idx = 0
        for r in range(rows):
            for c in range(0, columns, 2):
                region_image = y_hat_cpu_numpy[image_idx, :, :, 0]
                affinity_image = y_hat_cpu_numpy[image_idx, :, :, 1]
                ax[r, c].imshow(region_image, cmap="gray")
                ax[r, c].set_title("Region Map")
                ax[r, c+1].imshow(affinity_image, cmap="gray")
                ax[r, c+1].set_title("Affinity Map")
                image_idx += 1