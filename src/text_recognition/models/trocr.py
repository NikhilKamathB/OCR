import torch
from transformers import ViTModel, ViTConfig, \
                        TrOCRForCausalLM, TrOCRConfig, \
                        VisionEncoderDecoderModel, TrOCRProcessor


class TrOCRModel:

    def __init__(self, 
                 pretrained: bool = True,
                 pretrained_path: str = "microsoft/trocr-base-handwritten",
                 device: str = "mps",
                 processor: object = None,
                 processor_pretrained_path: str = "microsoft/trocr-base-handwritten",
                 max_length: int = 64,
                 early_stopping: bool = True,
                 no_repeat_ngram_size: int = 3,
                 length_penalty: float = 2.0,
                 num_beams: int = 4,
                ) -> None:
        encoder = ViTModel(ViTConfig())
        decoder = TrOCRForCausalLM(TrOCRConfig())
        if pretrained:
            self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_path)
        else:
            self.model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.processor = processor if processor is not None else TrOCRProcessor.from_pretrained(processor_pretrained_path)
        self.max_length = max_length
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        self.num_beams = num_beams

    def config(self) -> None:
        # declaration of special token
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        # declaration of vocabulary size
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        # declaration of beam search
        self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
        self.model.config.max_length = self.max_length
        self.model.config.early_stopping = self.early_stopping
        self.model.config.no_repeat_ngram_size = self.no_repeat_ngram_size
        self.model.config.length_penalty = self.length_penalty
        self.model.config.num_beams = self.num_beams