import torch.nn as nn

from loss import mse_loss
from datasets import SpectrogramDataset, TimeSeriesDataset

class SpectrVelCNNRegr(nn.Module):
    """Baseline model for regression to the velocity

    Use this to benchmark your model performance.
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=37120,out_features=1024)
        self.linear2=nn.Linear(in_features=1024,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)


class CombinedModel(nn.Module):
    """
    5 convolutional layers followed by 3 linear layers.
    In total 4.5M parameters.

    """
    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=12,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=12,
                        out_channels=24,
                        kernel_size=5,
                        stride=1,
                        padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=24,
                        out_channels=48,
                        kernel_size=5,
                        stride=1,
                        padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=48,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=3456,out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024,out_features=512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.linear_out=nn.Linear(in_features=256,out_features=1)

        self.flatten=nn.Flatten()
    
    def forward(self, input_data):
        x = self.conv_layer(input_data)
        x = self.flatten(x)
        x = self.linear_layer(x)
        return self.linear_out(x)


class SpectrVelLSTMRegr(nn.Module):
    """LSTM-based model for regression to predict golf ball velocity"""

    loss_fn = mse_loss  # Loss function
    dataset = TimeSeriesDataset  # Dataset to use

    def __init__(self):
        super().__init__()

        self.lstm_layer = nn.Sequential(
            nn.LSTM(input_size=2,
                    hidden_size=512,
                    num_layers=3,
                    batch_first=True,
                    dropout=0.3,
                    bidirectional=True),
        )

        self.attention = nn.Linear(512 * 2, 1)

        # Fully connected layers for regression
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=512 * 2, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

    def forward(self, input_data):
        # Reshape input_data from [batch_size, num_samples, seq_len, feature_dim]
        # to [batch_size * num_samples, seq_len, feature_dim]
        batch_size, num_samples, sequence_length, feature_dim = input_data.size()
        x = input_data.view(batch_size * num_samples, sequence_length, feature_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm_layer(x)  # lstm_out: [batch_size*num_samples, seq_len, hidden_size*2]

        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch_size*num_samples, seq_len, 1]
        context_vector = (attention_weights * lstm_out).sum(dim=1)  # Weighted sum over time

        # Residual connection: Combine raw LSTM output and attention output
        x = context_vector + lstm_out[:, -1, :]  # Last output as a shortcut

        # Pass through fully connected layers
        x = self.linear_layer(x)

        # Reshape output back to [batch_size, num_samples, -1]
        x = x.view(batch_size, num_samples, -1)

        # Average the outputs over the samples
        x = x.mean(dim=1)

        return x


class SimpleANN(nn.Module):
    """Simplified model with the ANN block shrunk 4 times.
        Around 10M parameters
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=37120,out_features=256)
        self.linear2=nn.Linear(in_features=256,out_features=64)
        self.linear3=nn.Linear(in_features=64,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)


class SimpleCNN(nn.Module):
    """Simplified model with the CNN block shrunk by a factor of 2.

    Around 20M parameters.
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=8,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=18560,out_features=1024)
        self.linear2=nn.Linear(in_features=1024,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)



class TeacherModel(nn.Module):
    """Teacher model for the Distilled Knowledge implementation.

    Same as baseline.
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=37120,out_features=1024)
        self.linear2=nn.Linear(in_features=1024,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/n**.5
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)



class StudentModel(nn.Module):
    """Model of the student. Used in the Distilled Knowledge part.

    Exactly 6651 parameters.
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=7,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=7,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=8,
                      out_channels=9,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=9,
                      out_channels=11,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=55,out_features=24)
        self.linear2=nn.Linear(in_features=24,out_features=5)
        self.linear3=nn.Linear(in_features=5,out_features=1)

        # Define activations once
        self.relu = nn.ReLU()
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        return x

    def _output_layer(self, x):
        return x

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)



# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule_stud(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/n**.5
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
