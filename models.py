import torch
import torchvision.models
import torch.nn as nn


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, n_of_classe):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, n_of_classe)
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x


class Resnet18(nn.Module):
    def __init__(self, n_of_classe):
        super(Resnet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, n_of_classe)
        )

    def forward(self, x):
        x = self.model(x)
        return x


aval_models = {'resnet18': Resnet18,
               'dino': DinoVisionTransformerClassifier}


def get_model(name, n_of_outs):
    assert name in aval_models, 'no models'
    model = aval_models[name](n_of_outs)
    return model
