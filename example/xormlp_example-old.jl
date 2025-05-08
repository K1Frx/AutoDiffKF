include("../src/mynn.jl")  # Załadowanie modułu mynn
using .Mynn: Linear, forward, Variable, ad_relu, backward, zero_grad!, ad_add, ad_tanh, ad_sigmoid, binary_cross_entropy, update_sgd, mse_loss

# Dane wejściowe XOR
X = [
    [-1.0, -1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [1.0, 1.0]
]

Y = [0.0, 1.0, 1.0, 0.0]

hidden_layer = Linear(2, 2, ad_relu)   # 2 wejścia, 2 neurony w warstwie ukrytej, funkcja aktywacji tanh
output_layer = Linear(2, 1, ad_sigmoid)    # 2 wejścia z warstwy ukrytej, 1 neuron wyjściowy

learning_rate = 0.3
epochs = 200

for epoch in 1:epochs
    total_loss = Variable(0.0)

    for i in 1:size(X, 1)

        zero_grad!(hidden_layer.weights)
        zero_grad!(hidden_layer.bias)
        zero_grad!(output_layer.weights)
        zero_grad!(output_layer.bias)

        x_input = Vector{Variable}([Variable(X[i][1]), Variable(X[i][2])])
        y_target = Y[i]
        
        hidden_output = forward(hidden_layer, x_input)
        y_pred = forward(output_layer, hidden_output)

        loss = mse_loss(y_pred, [y_target])
        total_loss = ad_add(total_loss, loss)

        backward(loss)

        update_sgd([hidden_layer.weights, hidden_layer.bias, output_layer.weights, output_layer.bias], learning_rate)
        
        zero_grad!(x_input)
        zero_grad!(hidden_output)
    end

end

println("\nTestowanie modelu:")
for i in 1:size(X, 1)
    x_input = [Variable(X[i][1]), Variable(X[i][2])]
    hidden_output = forward(hidden_layer, x_input)
    y_pred = forward(output_layer, hidden_output)
    println("Wejście: $(X[i, :]), Przewidywanie: $(y_pred[1].value), Oczekiwana wartość: $(Y[i])")
end