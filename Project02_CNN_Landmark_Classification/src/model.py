import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()
        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        # Conv - DropOut - BatchNorm - Activation - Pool
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,stride=1,kernel_size=5,padding=2,bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=16,out_channels=32,stride=1,kernel_size=5,padding=2,bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2,2), #112
            nn.Conv2d(in_channels=32,out_channels=64,stride=1,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2,2), #56
            nn.Conv2d(in_channels=64,out_channels=128,stride=1,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2,2), #28
            nn.Conv2d(in_channels=128,out_channels=256,stride=1,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2,2), #14
            nn.Conv2d(in_channels=256,out_channels=512,stride=1,kernel_size=3,padding=1,bias=False),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=14*14*512,out_features=512,bias=False),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(in_features=512,out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)

        x = self.model(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
