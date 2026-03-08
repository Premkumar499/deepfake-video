import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeModel(nn.Module):

    def __init__(self):

        super(DeepfakeModel, self).__init__()


        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.cnn = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        for param in self.cnn.parameters():
            param.requires_grad = False


        self.lstm = nn.LSTM(

            input_size=2048,  
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True

        )



        self.fc = nn.Sequential(

            nn.Linear(1024, 512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512, 2)

        )


    def forward(self, x):


        batch_size, seq_len, C, H, W = x.size()


        x = x.view(batch_size * seq_len, C, H, W)

        features = self.cnn(x)

        features = features.view(batch_size, seq_len, 2048)

        lstm_out, _ = self.lstm(features)

        final_output = lstm_out[:, -1, :]

        output = self.fc(final_output)

        return output