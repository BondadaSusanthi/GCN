import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.parameter import Parameter
from util import gen_A, gen_adj


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)  # (num_classes, out_features)
        output = torch.matmul(adj, support)
        return output


class GCNResNet(nn.Module):
    def __init__(self, num_classes, t, adj_file, in_channel=8):
        super(GCNResNet, self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, image_features, label_embeddings):
        feature = self.features(image_features)    # (B, 2048, 7, 7)
        feature = self.pooling(feature).view(feature.size(0), -1)  # (B, 2048)

        adj = gen_adj(self.A).detach()
        x = self.gc1(label_embeddings, adj)  # (num_classes, 1024)
        x = self.relu(x)
        x = self.gc2(x, adj)  # (num_classes, 2048)

        x = x.transpose(0, 1)  # (2048, num_classes)

        output = torch.matmul(feature, x)  # (B, num_classes)
        return output


def gcn_resnet101(num_classes, t, adj_file, in_channel=8):
    return GCNResNet(num_classes, t, adj_file, in_channel)
