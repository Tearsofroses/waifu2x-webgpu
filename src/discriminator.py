import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 96, 96)):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = 3
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(512, 1024, kernel_size=1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(1024, 1, kernel_size=1))
        
        # No Sigmoid here because we use BCEWithLogitsLoss for stability
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)