import warnings
from torch import nn
import torch

warnings.filterwarnings("ignore")

from transformers import (
    BertModel,
    BertConfig,
    DNATokenizer, BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

class GlobalModel(nn.Module):
    def __init__(self, model_path):
        super(GlobalModel, self).__init__()
        self.config = BertConfig.from_pretrained(model_path, finetuning_task="dnaprogram")
        self.bert = BertModel.from_pretrained(model_path, config=self.config)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=5, padding=2, stride=1),
            nn.ELU(),
            nn.BatchNorm1d(num_features=128)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2, stride=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_features=128)
        )
        self.dropout1 = nn.Dropout(p=0.3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=5, padding=2, stride=1)

        self.output = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=(1,)),
            nn.Flatten(start_dim=1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
