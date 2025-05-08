module Mymlp

include("Autodiffkf-old.jl")
using .Autodiff: ad_add, ad_mul, ad_relu, Variable, backward, zero_grad!, ad_tanh, ad_sigmoid, ad_log, ad_sub, ad_softmax, ad_div

export Linear, forward, mse_loss, update_sgd, Variable, backward, ad_relu, zero_grad!, ad_add, ad_tanh, ad_sigmoid, binary_cross_entropy, ad_softmax, ad_relu
export cross_entropy_loss, ad_div, update_adam, update_sgd

mutable struct Linear
    input_size::Int
    output_size::Int
    weights::Array{Variable, 2}
    bias::Array{Variable, 1}
    activation::Function
end

mutable struct Chain
    layers::Vector{Linear}
end

function DataLoader(x_train::Vector{Float64}, y_train::Vector{Float64}, batch_size::Int, shuffle::Bool=true)
    dataset = [(x_train[i], y_train[i]) for i in 1:length(x_train)]
    if shuffle
        shuffle!(dataset)
    end
    batches = [dataset[i:min(i + batch_size - 1, length(dataset))] for i in 1:batch_size:length(dataset)]
    return batches
end

function Chain(layers::Vector{Linear})
    return Chain(layers)
end

function Linear(input_size::Int, output_size::Int)
    weights = [Autodiff.Variable(randn() * sqrt(2 / (input_size + output_size))) for i in 1:output_size, j in 1:input_size]    
    bias = [Autodiff.Variable(0.0) for _ in 1:output_size]
    return Linear(input_size, output_size, weights, bias, ad_relu)
end

function Linear(input_size::Int, output_size::Int, activation::Function)
    weights = [Autodiff.Variable(randn() * sqrt(2 / (input_size + output_size))) for i in 1:output_size, j in 1:input_size]    
    bias = [Autodiff.Variable(0.0) for _ in 1:output_size]
    return Linear(input_size, output_size, weights, bias, activation)
end

function binary_cross_entropy(y_pred::Vector{Variable}, y_true::Vector{Float64})::Variable
    epsilon = 1e-12  # Mała wartość, aby uniknąć log(0)
    loss = Variable(0.0)
    for i in 1:length(y_pred)
        # Ograniczenie predykcji do zakresu [epsilon, 1 - epsilon]
        clipped_pred = ad_add(Variable(epsilon), ad_mul(Variable(1.0 - 2 * epsilon), y_pred[i]))
        
        # Obliczenie logarytmów
        log_pred = ad_log(clipped_pred)
        log_one_minus_pred = ad_log(ad_sub(Variable(1.0), clipped_pred))
        
        # Obliczenie straty dla jednej próbki
        loss = ad_add(loss, ad_add(
            ad_mul(Variable(-y_true[i]), log_pred),
            ad_mul(Variable(-(1.0 - y_true[i])), log_one_minus_pred)
        ))
    end
    # Zwrócenie średniej straty
    return ad_div(loss, Variable(length(y_pred)))
end

function cross_entropy_loss(y_pred::Vector{Variable}, y_true::Float64)::Variable
    loss = Variable(0.0)
    for i in 1:length(y_pred)
        if i == y_true + 1  # Correct index for the target class
            loss = ad_add(loss, ad_mul(Variable(-1.0), ad_log(y_pred[i])))
        end
    end
    return loss
end

function forward(layer::Linear, input::Vector{Variable})::Vector{Variable}
    results = Vector{Variable}()

    for i in 1:layer.output_size
        weighted_sum = Variable(0.0)
        for j in 1:layer.input_size
            weighted_sum = ad_add(weighted_sum, ad_mul(layer.weights[i, j], input[j]))
        end

        weighted_sum = ad_add(weighted_sum, layer.bias[i])
        push!(results, weighted_sum)
    end

    if layer.activation == ad_softmax
        return ad_softmax(results)
    else
        return [layer.activation(result) for result in results]
    end
end

function mse_loss(pred::Vector{Variable}, target::Vector{Float64})::Autodiff.Variable
    loss = Autodiff.Variable(0.0)
    for (p, t) in zip(pred, target)
        diff = ad_add(p, Autodiff.Variable(-t))
        loss = ad_add(loss, ad_mul(diff, diff))
    end
    return loss
end

function update_sgd(params::Vector{Array{Variable}}, learning_rate::Float64, clip_value::Union{Float64, Nothing}=nothing)
    if clip_value !== nothing
        for param in params
            for weight in param
                weight.grad = clamp(weight.grad, -clip_value, clip_value)
            end
        end
    end

    for param in params
        for weight in param
            weight.value -= learning_rate * weight.grad  # Aktualizacja wartości
        end
    end
end

function update_sgd(params::Vector{Array{Variable}}, learning_rate::Float64)
    update_sgd(params, learning_rate, nothing)
end

function update_adam()
    # Implementacja aktualizacji Adam
    # ...
end

end