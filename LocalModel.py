import warnings
from torch import nn
import torch

warnings.filterwarnings("ignore")


class LocalModel(nn.Module):
    def __init__(self, model_path):
        super(LocalModel, self).__init__()

        self.conv1_LocalFeature = nn.Sequential(
            nn.Conv1d(in_channels=224, out_channels=128, kernel_size=5, padding=2, stride=1),
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

        self.lstm = nn.LSTM(input_size=224, hidden_size=224, num_layers=1, batch_first=True, bidirectional=True)


        self.output = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=(1,)),
            nn.Flatten(start_dim=1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, token_type_ids):
        bert_data = self.bert(input_ids, token_type_ids=token_type_ids)
        bert_data = bert_data[0]
        lstm1, (hd, cn) = self.lstm(bert_data)
        lstm1 = lstm1[:, :, :224] + lstm1[:, :, 224:]

        conv1_seq = self.conv1_LocalFeature(lstm1.transpose(1, 2))
        maxpool1 = self.maxpool1(conv1_seq)
        maxpool1 = self.dropout1(maxpool1)

        conv2_seq = self.conv2(maxpool1)
        output1 = self.output(conv2_seq)

        return output1

