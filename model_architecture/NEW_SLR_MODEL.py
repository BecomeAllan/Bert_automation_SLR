from torchmetrics import Recall, Accuracy , F1, AUROC
import pytorch_lightning as pl

class Train(pl.LightningModule):
  def __init__(self):
    super(Train, self).__init__()

    self.accuracy = Accuracy()
    self.R = Recall()
    self.F1 = F1()
    self.AUROC = AUROC(num_classes=2)

    self.table = []
    self.losses_train = []
    self.losses_val = []


  def training_step(self, batch, batch_idx):
    opt_a = self.optimizers()

    # the is batch = dict(labels, input_ids, attention_mask)
    labels = batch.pop("labels", None) 
    
    # FOWARD(...)
    predict = self(**batch)

    loss = 0
    if labels is not None:
        loss = self.criterion(predict, labels)


    self.log("train_loss", loss )
    return {"loss": loss, "predictions": predict.detach().cpu(), "labels": labels.detach().cpu()}    


  def training_epoch_end(self, outputs):
    labels = []
    predictions = []
    loss = []

    for output in outputs:
      labels.append(output["labels"])
      predictions.append(output["predictions"])
      loss.append(output["loss"])

    predictions = torch.squeeze(torch.cat(predictions, dim=0))
    labels = torch.squeeze(torch.cat(labels, dim=0))
    loss = torch.mean(torch.tensor(loss).detach().cpu())

    self.losses_train.append(loss)
    auc = self.AUROC(predictions , labels)

    predictions = torch.argmax(predictions, 1)
    
    a = {"Epoch": self.current_epoch,
           "Train_Acc": round(self.accuracy(predictions,labels).item(),4),
           "Train_loss": round(loss.item(),4),
           "Train_R": round(self.R(predictions, labels).item(),4),
          #  "Train_F1": round(self.F1(predictions, labels).item(),4),
           "Train_AUROC": round(auc.item(),4),
          #  "Train_WSS": round(wss(labels,predictions),4),
           "Train_WSS@R": round(wssR(labels,predictions),4),
           }

    self.table.insert(0, a)

    print(f"{'Epoch ' + str(self.current_epoch):^9} | {'Loss':^12} | {'Acc':^10} | {'AUROC':^9} | {'WSS@R':^9} | {'R':^9}")
    print(f"{'train':^9} | {self.table[0]['Train_loss']:^12} | {self.table[0]['Train_Acc']:^10} | {self.table[0]['Train_AUROC']:^9} | {self.table[0]['Train_WSS@R']:^9} | {self.table[0]['Train_R']:^9}")
    print(f"{'Val':^9} | {self.table[1]['Val_loss']:^12} | {self.table[1]['Val_Acc']:^10} | {self.table[1]['Val_AUROC']:^9} | {self.table[1]['Val_WSS@R']:^9} | {self.table[1]['Val_R']:^9}")
    print("---"*30)

    self.table = []
    
  def on_train_end(self):
    plot(self.losses_train, self.losses_val, self.current_epoch)
    self.losses_train = self.losses_val = []

  def validation_step(self, batch, batch_idx):

    labels = batch.pop("labels", None) 
    predict = self(**batch)

    loss = 0
    if labels is not None:
        loss = self.criterion(predict, labels)

    self.log("val_loss", loss)
    return {"loss": loss, "predictions": predict.detach().cpu(), "labels": labels.detach().cpu()}
    
  def validation_epoch_end(self, outputs):

    labels = []
    predictions = []
    loss = []

    for output in outputs:
      labels.append(output["labels"].detach().cpu())
      predictions.append(output["predictions"])
      loss.append(output["loss"])

    predictions = torch.squeeze(torch.cat(predictions, dim=0))
    labels = torch.squeeze(torch.cat(labels, dim=0))
    loss = torch.mean(torch.tensor(loss).detach().cpu())
    
    self.losses_val.append(loss)
    
    auc = self.AUROC(predictions , labels)
    predictions = torch.argmax(predictions, 1)
    
    a = {"Epoch": self.current_epoch,
           "Val_Acc": round(self.accuracy(predictions,labels).item(),4),
           "Val_loss": round(loss.item(),4),
           "Val_R": round(self.R(predictions, labels).item(),4),
          #  "Val_F1": round(self.F1(predictions, labels).item(),4),
           "Val_AUROC": round(auc.item(),4),
          #  "Val_WSS": round(wss( labels,predictions),4),
           "Val_WSS@R": round(wssR(labels,predictions),4),
           }

    self.table.insert(0, a)

  def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):

    if self.output_attentions:
      labels = batch.pop('labels')
      bert_output = self.bert(**batch)
      predict = self.classifier(bert_output.pooler_output)
      

      return {"predictions": torch.squeeze(predict, 0),
              "attentions": bert_output.attentions,
              "labels": torch.squeeze(labels,0),
              "input_ids": torch.squeeze(batch['input_ids'],0),
              }

    else:
      predict = self(**batch)
      return {"predictions": torch.squeeze(predict, 0), "labels": torch.squeeze(batch["labels"],0)}


class Model_SLR(Train):

  def __init__(self, lr = 2e-5,
               n_training_steps=1,
               freeze_bert=True,
               output_attentions = False ,
               n_warmup_steps=1,
               **data):
    
    super(Model_SLR, self).__init__()
    # self.automatic_optimization = False
    pl.seed_everything(data.get('seed', 1))
    torch.manual_seed(data.get('seed', 1))
    # print("SEED:", data.get('seed', 1))

    self.output_attentions = output_attentions

    self.bert = BertModel.from_pretrained(BERT_MODEL_NAME,
                                          return_dict=True,
                                          output_attentions = output_attentions)

    self.criterion = nn.CrossEntropyLoss(reduction = 'mean')

    # Freeze the BERT model
    if freeze_bert:
      for param in self.bert.parameters():
        param.requires_grad = freeze_bert
    
    bert_layers = list(self.bert.children())


    layers = data.get('n_encode_layers', 12)

    bert_layers_encode = torch.nn.Sequential(*(list(bert_layers[1].layer.children())[0:layers]))

    self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.bert.config.hidden_size),
            nn.Dropout(data.get("drop", 0.5)),
            nn.Linear(self.bert.config.hidden_size, 2),
            nn.Softmax(1),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(10, 1),
            # nn.Sigmoid(),
        )
    
    self.bert = torch.nn.Sequential(bert_layers[0],
                                    bert_layers_encode,
                                    bert_layers[2])

    nn.init.normal_(self.classifier[2].weight, mean=0, std=0.00001)
    nn.init.zeros_(self.classifier[2].bias)
    
    
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.lr = lr



  def forward(self, input_ids, attention_mask, **args):

    bert_output = self.bert(input_ids, attention_mask=attention_mask)
    predict = self.classifier(bert_output.pooler_output)

    return predict

  def configure_optimizers(self):

    optimizer = AdamW(self.parameters(), lr=self.lr)

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )

    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )