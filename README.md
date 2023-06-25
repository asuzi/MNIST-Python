<h1>MNIST</h1>
My take on Machine Learning using the popular MNIST database.<br>
The goal is to predict what number is written on a 28x28 pixel image. The numbers range from 0 to 9.

In this example I implement a Linear Neural Network and achieved an accuracy rate of 96.5%

The Model looks like this.<br>
<i>NeuralNetwork( <br>
  (flatten): Flatten(start_dim=1, end_dim=-1)<br>
  (linear_relu_stack): Sequential(<br>
    (0): Linear(in_features=784, out_features=512, bias=True)<br>
    (1): ReLU()<br>
    (2): Linear(in_features=512, out_features=512, bias=True)<br>
    (3): ReLU()<br>
    (4): Linear(in_features=512, out_features=10, bias=True)<br>
  )<br>
)<br></i>


<b>Model.py</b> <br>
Here the model is constructed, trained and tested. See simplified steps below.
1. Read the given CSV file and turn it into PyTorch compatible tensors.
2. Begin training with given data for x amount of epochs.
3. After every training epoch test the accuracy on testing dataset.
4. Save the most successful model in to a file.

>>Model.py
Output: 

Training 1 epoch out of 3.
 ------------------
loss: 2.309494  [   64/60000]<br>
loss: 0.627912  [ 6464/60000]<br>
loss: 0.366829  [12864/60000]<br>
loss: 0.372798  [19264/60000]<br>
loss: 0.250612  [25664/60000]<br>
loss: 0.322473  [32064/60000]<br>
loss: 0.221325  [38464/60000]<br>
loss: 0.291895  [44864/60000]<br>
loss: 0.281976  [51264/60000]<br>
loss: 0.278504  [57664/60000]<br>
Test results: <br>
 Accuracy: 93.6%, Avg loss: 0.213179<br>

Training 2 epoch out of 3.
 ------------------
loss: 0.118671  [   64/60000]<br>
loss: 0.189939  [ 6464/60000]<br>
loss: 0.122716  [12864/60000]<br>
loss: 0.281849  [19264/60000]<br>
loss: 0.103203  [25664/60000]<br>
loss: 0.226752  [32064/60000]<br>
loss: 0.108506  [38464/60000]<br>
loss: 0.231444  [44864/60000]<br>
loss: 0.177283  [51264/60000]<br>
loss: 0.185972  [57664/60000]<br>
Test results: <br>
 Accuracy: 95.5%, Avg loss: 0.142301<br>

Training 3 epoch out of 3.
 ------------------
loss: 0.066214  [   64/60000]<br>
loss: 0.140154  [ 6464/60000]<br>
loss: 0.099441  [12864/60000]<br>
loss: 0.192799  [19264/60000]<br>
loss: 0.064127  [25664/60000]<br>
loss: 0.168688  [32064/60000]<br>
loss: 0.079495  [38464/60000]<br>
loss: 0.183989  [44864/60000]<br>
loss: 0.132687  [51264/60000]<br>
loss: 0.145497  [57664/60000]<br>
Test results: <br>
 Accuracy: 96.5%, Avg loss: 0.108685<br>
<br>
<b>Load_Model.py</b><br>
This is not necessary, but here we will load the previously saved model and run tests with it.<br>
To load the data I'm using PyTorch built-in datasets function to load the MNIST testing data.<br>
Using this I would be able to implement the AI to label any drawn pictures of numbers from 0 to 9<br>
<br>
#Loops 20 examples.

>>Load_Model.py<br>
Output: <br>
<br>
Predicted: "7", Actual: "7"<br>
Predicted: "2", Actual: "2"<br>
Predicted: "1", Actual: "1"<br>
Predicted: "0", Actual: "0"<br>
Predicted: "4", Actual: "4"<br>
Predicted: "1", Actual: "1"<br>
Predicted: "4", Actual: "4"<br>
Predicted: "9", Actual: "9"<br>
Predicted: "5", Actual: "5"<br>
Predicted: "9", Actual: "9"<br>
Predicted: "0", Actual: "0"<br>
Predicted: "6", Actual: "6"<br>
Predicted: "9", Actual: "9"<br>
Predicted: "0", Actual: "0"<br>
Predicted: "1", Actual: "1"<br>
Predicted: "5", Actual: "5"<br>
Predicted: "9", Actual: "9"<br>
Predicted: "7", Actual: "7"<br>
Predicted: "3", Actual: "3"<br>
Predicted: "4", Actual: "4"<br>
<br>
<b>
Sources:<br>
https://pytorch.org/docs/stable/index.html <br>
Dataset:<br>
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
</b>













