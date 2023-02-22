import tensorflow
import torch


model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

scaler = GradScaler()

for epoch in epochs:
	for input, target in data:
		optimizer.zero_grad()

		with autocast(device_type='cuda', dtype=torch.float16):
			output = model(input)
			loss = loss_fn(output, target)

		scaler.scale(loss).backward()
		scaler.step(optimizer)

		scaler.update()
