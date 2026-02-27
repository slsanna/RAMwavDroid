import torch.nn as nn

class SampleCNN(nn.Module):
    DROPOUT = 0.5
    def __init__(self, numClasses=5):
        super(SampleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        
        self.conv7 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm1d(256)
        
        self.conv8 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm1d(256)
        
        self.conv9 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm1d(256)
        
        self.conv10 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm1d(512)
        
        self.conv11 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm1d(512)
        
        self.maxpool = nn.MaxPool1d(3, stride=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.DROPOUT)
        
        self.fc = nn.Linear(512, numClasses)

    def forward(self, inputTensor):
        if inputTensor.ndim == 2:
            inputTensor = inputTensor.unsqueeze(1)
        
        output = self.relu(self.bn1(self.conv1(inputTensor)))
        
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.maxpool(output)
        
        output = self.relu(self.bn3(self.conv3(output)))
        output = self.maxpool(output)
        
        output = self.relu(self.bn4(self.conv4(output)))
        output = self.maxpool(output)
        
        output = self.relu(self.bn5(self.conv5(output)))
        output = self.maxpool(output)
        
        output = self.relu(self.bn6(self.conv6(output)))
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = self.relu(self.bn7(self.conv7(output)))
        output = self.maxpool(output)
        
        output = self.relu(self.bn8(self.conv8(output)))
        output = self.maxpool(output)
        
        output = self.relu(self.bn9(self.conv9(output)))
        output = self.maxpool(output)
        
        output = self.relu(self.bn10(self.conv10(output)))
        output = self.maxpool(output)
        
        output = self.relu(self.bn11(self.conv11(output)))
        output = self.dropout(output)
        
        output = output.view(output.size(0), -1)
        logits = self.fc(output)
        return logits