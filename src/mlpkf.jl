module Mlpkf

include("autodiffkf.jl")
using .Autodiffkf: Variable, +, -, *, /, exp, log, relu, sigmoid, tanh, softmax, zero_grad!, backward, print_gradients, min, max

export Linear, Chain, Dataset, train_epoch, predict_dataset, initialize_data, mse_loss, sgd
export exp, log, relu, sigmoid, tanh, softmax, binary_crossentropy, adam

mutable struct Linear
    input_size::Int
    output_size::Int
    weights::Array{Variable, 2}
    bias::Array{Variable, 1}
    activation::Function
end

mutable struct Chain
    hidden_layers::Vector{Linear}
    output_layer::Linear
end

mutable struct Dataset
    inputs::Vector{Vector{Variable}}
    targets::Vector{Variable}
end

function train_epoch(
    dataset::Dataset,
    model::Chain,
    loss_function::Function,
    optimizer::Function,
    learning_rate::Float64
)::Float64
    total_loss = Variable(0.0)
    num_samples = length(dataset.inputs)
    output::Vector{Variable} = []
    for (input, target) in zip(dataset.inputs, dataset.targets)
        for layer in model.hidden_layers
            zero_grad!(layer.weights)
            zero_grad!(layer.bias)
        end
        zero_grad!(model.output_layer.weights)
        zero_grad!(model.output_layer.bias)
        zero_grad!(input)

        output = input
        for layer in model.hidden_layers
            output = forward(layer, output)
        end
        output = forward(model.output_layer, output)

        loss = loss_function(output, target)
        total_loss = total_loss + loss

        backward(loss)
        
        optimizer(model, learning_rate)
    end

    avg_loss = total_loss.value / num_samples

    return avg_loss
end

function predict_dataset(
    dataset::Dataset,
    model::Chain
)::Vector{Float64}
    predictions = Vector{Float64}()

    for input in dataset.inputs
        output = input
        for layer in model.hidden_layers
            output = forward(layer, output)
        end
        output = forward(model.output_layer, output)
        push!(predictions, output[1].value)
    end

    return predictions
end

function initialize_data(X::Matrix{Float64}, Y::Vector{Float64})::Dataset
    inputs = [Vector{Variable}([Variable(x) for x in row]) for row in eachrow(X)]
    targets = [Variable(y) for y in Y]
    return Dataset(inputs, targets)
end

function initialize_data(X::Union{Matrix{Float64}, Vector{Vector{Variable}}}, Y::Vector{Float64}, batch_size::Int)::Vector{Dataset}
    dataset = initialize_data(X, Y)
    batches = []
    num_samples = length(dataset.inputs)
    for i in 1:batch_size:num_samples
        batch_inputs = dataset.inputs[i:min(i + batch_size - 1, num_samples)]
        batch_targets = dataset.targets[i:min(i + batch_size - 1, num_samples)]
        push!(batches, Dataset(batch_inputs, batch_targets))
    end
    return batches
end

function initialize_data(X::Vector{Vector{Float64}}, Y::Vector{Float64})::Dataset
    inputs = [Vector{Variable}([Variable(x) for x in row]) for row in X]
    targets = [Variable(y) for y in Y]
    return Dataset(inputs, targets)
end

function Linear(input_size::Int, output_size::Int)
    weights = [Variable(randn() * sqrt(2 / (input_size + output_size))) for i in 1:output_size, j in 1:input_size]    
    bias = [Variable(0.0) for _ in 1:output_size]
    return Linear(input_size, output_size, weights, bias, relu)
end

function Linear(input_size::Int, output_size::Int, activation::Function)
    weights = []
    if activation == relu
        weights = [Variable(randn() * sqrt(2 / (input_size + output_size))) for i in 1:output_size, j in 1:input_size]
    elseif activation == sigmoid || activation == tanh
        weights = [Variable(randn() * sqrt(1 / input_size)) for i in 1:output_size, j in 1:input_size]
    end
    bias = [Variable(0.0) for _ in 1:output_size]
    return Linear(input_size, output_size, weights, bias, activation)
end

function Chain(input_layer::Linear, output_layer::Linear)
    return Chain([input_layer], output_layer)
end

function forward(layer::Linear, input::Vector{Variable})::Vector{Variable}
    results = Vector{Variable}()

    for i in 1:layer.output_size
        weighted_sum = Variable(0.0)
        for j in 1:layer.input_size
            weighted_sum = weighted_sum + (layer.weights[i, j] * input[j])
        end
        
        weighted_sum = weighted_sum + layer.bias[i]
        push!(results, weighted_sum)
    end

    if layer.activation == softmax
        return softmax(results)
    else
        return [layer.activation(result) for result in results]
    end
end

function mse_loss(pred::Vector{Variable}, target::Variable)
    loss = Variable(0.0)
    for p in pred
        diff = p - target
        loss = loss + (diff * diff)
    end
    return loss
end

function binary_crossentropy(pred::Vector{Variable}, target::Variable)::Variable
    epsilon = Variable(1e-12)
    loss = Variable(0.0)
    for p in pred
        clipped_pred = max(min(p, Variable(1.0) - epsilon), epsilon)
        loss = loss - (target * log(clipped_pred) + (Variable(1.0) - target) * log(Variable(1.0) - clipped_pred))
    end
    return loss
end

function sgd(model::Chain, learning_rate::Float64)
    for layer in model.hidden_layers
        for w in layer.weights
            w.value -= learning_rate * w.grad
        end
        for b in layer.bias
            b.value -= learning_rate * b.grad
        end
    end
    for w in model.output_layer.weights
        w.value -= learning_rate * w.grad
    end
    for b in model.output_layer.bias
        b.value -= learning_rate * b.grad
    end
end

function adam(model::Chain, learning_rate::Float64, beta1::Float64 = 0.9, beta2::Float64 = 0.999, epsilon::Float64 = 1e-8, t::Int = 1)
    for layer in model.hidden_layers
        for w in layer.weights
            w.m = beta1 * w.m + (1 - beta1) * w.grad
            w.v = beta2 * w.v + (1 - beta2) * w.grad^2

            m_hat = w.m / (1 - beta1^t)
            v_hat = w.v / (1 - beta2^t)

            w.value -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        end
        for b in layer.bias
            b.m = beta1 * b.m + (1 - beta1) * b.grad
            b.v = beta2 * b.v + (1 - beta2) * b.grad^2

            m_hat = b.m / (1 - beta1^t)
            v_hat = b.v / (1 - beta2^t)

            b.value -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        end
    end

    for w in model.output_layer.weights
        w.m = beta1 * w.m + (1 - beta1) * w.grad
        w.v = beta2 * w.v + (1 - beta2) * w.grad^2

        m_hat = w.m / (1 - beta1^t)
        v_hat = w.v / (1 - beta2^t)

        w.value -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    end

    for b in model.output_layer.bias
        b.m = beta1 * b.m + (1 - beta1) * b.grad
        b.v = beta2 * b.v + (1 - beta2) * b.grad^2

        m_hat = b.m / (1 - beta1^t)
        v_hat = b.v / (1 - beta2^t)

        b.value -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    end
end

end