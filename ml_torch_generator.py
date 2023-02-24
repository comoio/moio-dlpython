import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TheModelClass(nn.Module):
	def __init__(self):
		super(TheModelClass, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 3)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

model = TheModelClass()


torch.manual_seed(1)
#<torch._C.Generator at 0x23daae3dfb0>

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [6], [9]])

W = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)
hypothesis = x_train * W + b

cost = torch.mean((hypothesis -y_train) ** 2)
optimizer = optim.SGD([W, b], lr = 0.01)

nb_epochs = 3000

for epoch in range(nb_epochs + 1):
	hypothesis = x_train * W + b
	cost = torch.mean((hypothesis - y_train) ** 2)
	optimizer.zero_grad()
	cost.backward()
	optimizer.step()

	if epoch % 100 == 0:
		print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {: .6f}'.format(epoch, nb_epochs, W.item(), b.item(), cost.item()))

torch.save(model.state_dict(), './model.pt')

model.load_state_dict(torch.load('./model.pt'))
model.eval()
