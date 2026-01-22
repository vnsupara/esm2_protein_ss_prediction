import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=480, hidden=256, num_classes=9):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden*2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,480)
        out,_ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
