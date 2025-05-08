include("../src/Mlpkf.jl")
using .Mlpkf: Linear, Chain, Dataset, train_epoch, predict_dataset, initialize_data, mse_loss, sgd
using .Mlpkf: exp, log, relu, sigmoid, tanh, softmax

# Dane wejściowe XOR
X = [
    [-1.0, -1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [1.0, 1.0]
]
Y = [0.0, 1.0, 1.0, 0.0]

dataset = initialize_data(X, Y)

model = Chain(
    Linear(2, 2, relu),
    Linear(2, 1, sigmoid)
)

epochs = 500
learning_rate = 0.1

for epoch in 1:epochs
    loss = train_epoch(dataset, model, mse_loss, sgd, learning_rate)
    println("Epoch $epoch: Loss = $loss")
end

println("\nTestowanie modelu:")
predictions = predict_dataset(dataset, model)
for (input, target, prediction) in zip(dataset.inputs, dataset.targets, predictions)
    println("Przewidywanie: $(prediction), Oczekiwana wartość: $(target.value)")
end