import torch


x = torch.tensor(4.0)
target = torch.tensor(12.0)

weight = torch.tensor(
    1.0,
    requires_grad=True,
)

learning_rate = 0.01


for epoch in range(30):
    prediction = x * weight
    loss = (prediction - target) ** 2

    loss.backward()

    assert weight.grad is not None

    with torch.no_grad():
        weight -= learning_rate * weight.grad

    weight.grad.zero_()


# Calculate loss using the final updated weight
prediction = x * weight
loss = (prediction - target) ** 2

print("Final weight:", weight.item())
print("Final prediction:", prediction.item())
print("Final loss:", loss.item())
