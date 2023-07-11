import pytorch_lightning as pl
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
import torch.optim as optim


class T5Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        # self.rouge_metric = load_metric("rouge")

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output.loss, output.logits

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        loss, outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
        )
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, outputs = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs = self._step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def genrate_summary(self, text):
        input_ids = self.tokenizer.encode(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        generated_ids = self.model.generate(
            input_ids,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
