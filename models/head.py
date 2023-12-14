import torch 

class Head_Cls(torch.nn.Module):
    def __init__(self, in_features=512, out_features=1) -> None:
        super().__init__()
        self.middle = torch.nn.Linear(in_features, 128)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.1)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.qs = torch.nn.Linear(128, out_features)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.middle(x)
        x = self.leaky(x)
        x = self.dropout(x)
        return self.qs(x)

