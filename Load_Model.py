import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load the dataset.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Select device for training process. (CPU or CUDA). 
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# The model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# Define Model and load model with the pre-trained model.
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('C:\\path\\to\\the_saved_data.csv'))

# All possible predictable classes
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

# Run the testing data through the model and make predictions!
model.eval()
for i in range(20):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

