import torch.nn as nn
import timm

n_classes = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        try:
            # The model is a Vision transformer
            self.model = timm.create_model('vit_large_patch16_384', pretrained=True)
        except:
            print("Failed to load models")
            raise SystemExit(0)

        for param in self.model.parameters():
            param.requires_grad = False

        for i in [-1,-2, -3]:
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True  

        self.in_features = self.model.head.in_features
        self.model.head = nn.Linear(self.in_features, self.in_features//5)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.in_features//5, n_classes)
          
    def forward(self, x):
        x = self.model(x)        
        x = self.dropout(self.relu(x))

        return self.fc(x)
