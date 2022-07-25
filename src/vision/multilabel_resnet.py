from statistics import mode
import torch
import torch.nn as nn
from torchvision.models import resnet18


class MultilabelResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Consider which activation function to use
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.activation = nn.Sigmoid()
        self.loss_criterion = nn.BCELoss(reduction = 'mean')    

        ############################################################################
        # Student code begin
        ############################################################################

        output_classes = 7

        model = resnet18(pretrained = True)

        
        self.conv_layers = nn.Sequential(*list(model.children())[:-1])
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_classes, bias=True)
        )

        ############################################################################
        # Student code end
        ############################################################################

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
        ############################################################################
        # Student code begin
        ############################################################################
        # changed
        output = self.conv_layers(x)
        output = output.view(x.size(0), -1)
        y = self.fc_layers(output)
        model_output = self.activation(y)
        # print(x.shape)
        # y = torch.flatten(x, start_dim = 2) 
        # z = self.fc_layers(y)
        # model_output = self.activation(z)


        # model_output = model_output.reshape((x.shape[0], 7))
        # print(model_output.shape)
        
        # print(model_output)
        # torch.autograd.set_detect_anomaly(True)
        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`multi_resnet.py` needs to be implemented"
        # )


        ############################################################################
        # Student code end
        ############################################################################
        return model_output
