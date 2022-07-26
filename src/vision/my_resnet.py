import torch
import torch.nn as nn
from torchvision.models import resnet18

# class MyModel(nn.Module):
#     def __init__(self, pretrained_model):
#         self.pretrained_model = pretrained_model
#         self.last_layer = ... # create layer

#     def forward(self, x):
#         return self.last_layer(self.pretrained_model(x))


class MyResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')


        output_classes = 15

        model = resnet18(pretrained = True)
        # self.fc_layers = model.fc.parameters()
        # for (name, layer) in model._modules.items():
        #     #iteration over outer layers
        #     print((name, layer))

        # print('Children')
        # print(list(model.children()))
        # print('Modules')
        # print(list(model.modules()))  

        for param in model.parameters():
            param.requires_grad = False
        
        self.conv_layers = nn.Sequential(*list(model.children())[:-1])
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=512, out_features=15, bias=True)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim = 1)
        model_output = self.fc_layers(x)

        return model_output
