import torch
import torch.nn as nn
import torchvision.models as models


class SimpleBaseline(nn.Module):
    """
    Simple Baseline for Human Pose Estimation as described in
    "Simple Baselines for Human Pose Estimation and Tracking" (Xiao et al., 2018)

    This implementation uses ResNet-50 backbone for faster inference.
    """
    def __init__(self, num_keypoints=17, deconv_with_bias=False):
        """
        Args:
            num_keypoints (int): Number of keypoints to predict
            deconv_with_bias (bool): Whether to use bias in deconvolution layers
        """
        super(SimpleBaseline, self).__init__()

        # Initialize ResNet-50 backbone for faster inference
        self.backbone = models.resnet50(pretrained=True)

        # Remove the last two layers: avgpool and fc
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Deconvolution layers
        self.deconv_layers = self._make_deconv_layers(
            3,                  # Number of deconv layers
            [256, 256, 256],    # Number of filters per layer
            [4, 4, 4],          # Kernel size for each layer
            deconv_with_bias
        )

        # Final layer to predict heatmaps
        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layers(self, num_layers, num_filters, kernel_sizes, bias):
        """Create the deconvolution layers as described in the paper"""
        layers = []

        # Get the number of channels from the backbone
        inplanes = 2048  # For ResNet50/101/152, final stage has 2048 channels

        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=num_filters[i],
                    kernel_size=kernel_sizes[i],
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=bias
                )
            )
            layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            inplanes = num_filters[i]

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network"""
        # Forward pass through the backbone (ResNet)
        x = self.backbone(x)

        # Forward pass through the deconvolution layers
        x = self.deconv_layers(x)

        # Final layer to produce heatmaps
        x = self.final_layer(x)

        return x

def create_model(num_keypoints=42):
    """
    Create a SimpleBaseline model with ResNet-50 backbone
    Args:
        num_keypoints (int): Number of keypoints to predict
    Returns:
        SimpleBaseline model with ResNet-50 backbone
    """
    model = SimpleBaseline(num_keypoints=num_keypoints)
    return model

# Example usage
if __name__ == "__main__":
    # Create model with ResNet-50 backbone
    model = create_model()

    # Check model structure
    print(model)

    # Generate a sample input
    sample_input = torch.randn(2, 3, 256, 192)  # [batch_size, channels, height, width]

    # Forward pass
    output = model(sample_input)

    # Print output shape
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [2, 17, 64, 48] for default settings
