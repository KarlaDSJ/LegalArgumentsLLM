import torch
import torch.nn as nn

class LegalClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=2): #256
        super(LegalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)#0.3
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
