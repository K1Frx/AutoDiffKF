module Autodiffkf
using Base.Math

export Variable, ad_add, ad_mul, ad_sin, ad_cos, ad_log, ad_exp, backward, ad_div

mutable struct Variable
    value::Float64
    grad::Float64
    parents::Vector{Any}
    op_type::Symbol
end

function Variable(value::Float64)
    return Variable(value, 0.0, [], :none)
end

function Variable(value::Float32)
    return Variable(Float64(value), 0.0, [], :none)
end

function Variable(value::Int)
    return Variable(Float64(value), 0.0, [], :none)
end

function Variable(value::Float64, symbol::Symbol)
    return Variable(value, 0.0, [], symbol)
end

function ReadableVariable(x::Variable)
    return x.value
end

function ad_add(a::Variable, b::Variable)
    result = Variable(a.value + b.value, :add)
    push!(result.parents, (a, b))

    return result
end

function ad_mul(a::Variable, b::Variable)
    result = Variable(a.value * b.value, :mul)
    push!(result.parents, (a, b))
    return result
end

function ad_sub(a::Variable, b::Variable)
    result = Variable(a.value - b.value, :sub)
    push!(result.parents, (a, b))
    return result
end

function ad_sin(a::Variable)
    result = Variable(sin(a.value), :sin)
    push!(result.parents, a)
    return result
end

function ad_cos(a::Variable)
    result = Variable(cos(a.value), :cos)
    push!(result.parents, a)
    return result
end

function ad_exp(a::Variable)
    result = Variable(exp(a.value), :exp)
    push!(result.parents, a)
    return result
end

function ad_log(a::Variable)
    result = Variable(log(a.value), :log)
    push!(result.parents, a)
    return result
end

function ad_relu(a::Variable)
    result = Variable(max(0.0, a.value), :relu)
    push!(result.parents, a)
    return result
end

function ad_sigmoid(x::Variable)::Variable
    raw_sigmoid = ad_div(Variable(1.0), ad_add(Variable(1.0), ad_exp(ad_sub(Variable(0.0), x))))
    result = Variable(raw_sigmoid.value, :sigmoid)
    push!(result.parents, x)  # Dodanie rodzica do grafu obliczeniowego
    return result
end

function ad_tanh(a::Variable)
    result = Variable(tanh(a.value), :tanh)
    push!(result.parents, a)
    return result
end

function ad_div(a::Variable, b::Variable)
    result = Variable(a.value / b.value, :div)
    push!(result.parents, (a, b))
    return result
end

function zero_grad!(v::Variable)
    v.grad = 0.0
    for parent in v.parents
        if v.op_type in [:add, :mul]
            a, b = parent
            zero_grad!(a)
            zero_grad!(b)
        else
            zero_grad!(parent)
        end
    end
end

function ad_softmax(inputs::Vector{Variable})::Vector{Variable}
    epsilon = 1e-12  # Mała wartość, aby uniknąć log(0)
    exp_values = [ad_exp(input) for input in inputs]
    sum_exp = Variable(0.0)
    for value in exp_values
        sum_exp = ad_add(sum_exp, value)
    end
    return [ad_div(ad_add(value, Variable(epsilon)), ad_add(sum_exp, Variable(epsilon))) for value in exp_values]
end

function zero_grad!(v::Matrix{Variable})
    for row in eachrow(v)
        for element in row
            zero_grad!(element)
        end
    end
end

function zero_grad!(v::Vector{Variable})
    for element in v
        zero_grad!(element)
    end
end

function backward(v::Variable, visited::Set{Variable}=Set{Variable}())
    if v in visited && all(parent in visited for parent in v.parents)
        return
    end

    if v ∉ visited && v.grad == 0.0
        v.grad = 1.0
    end

    push!(visited, v)

    for parent in v.parents
        if v.op_type == :add
            a, b = parent
            if a !== nothing
                a.grad += v.grad * 1
                push!(visited, a)
                backward(a, visited)
            end
            if b !== nothing
                b.grad += v.grad * 1
                push!(visited, b)
                backward(b, visited)
            end

        elseif v.op_type == :mul
            a, b = parent
            if a !== nothing
                a.grad += v.grad * b.value
                push!(visited, a)
                backward(a, visited)
            end
            if b !== nothing
                b.grad += v.grad * a.value
                push!(visited, b)
                backward(b, visited)
            end

        elseif v.op_type == :sin
            a = parent
            if a !== nothing
                a.grad += v.grad * cos(a.value)
                push!(visited, a)
                backward(a, visited)
            end

        elseif v.op_type == :cos
            a = parent
            if a !== nothing
                a.grad += v.grad * -sin(a.value)
                push!(visited, a)
                backward(a, visited)
            end

        elseif v.op_type == :log
            a = parent
            if a !== nothing
                a.grad += v.grad * (1 / a.value)
                push!(visited, a)
                backward(a, visited)
            end

        elseif v.op_type == :exp
            a = parent
            if a !== nothing
                a.grad += v.grad * exp(a.value)
                push!(visited, a)
                backward(a, visited)
            end

        elseif v.op_type == :relu
            a = parent
            if a !== nothing
                a.grad += v.grad * (a.value > 0.0 ? 1.0 : 0.0)
                push!(visited, a)
                backward(a, visited)
            end

        elseif v.op_type == :sigmoid
            a = parent
            if a !== nothing
                a.grad += v.grad * (v.value * (1.0 - v.value))
                push!(visited, a)
                backward(a, visited)
            end

        elseif v.op_type == :tanh
            a = parent
            if a !== nothing
                a.grad += v.grad * (1 - tanh(a.value)^2)
                push!(visited, a)
                backward(a, visited)
            end

        elseif v.op_type == :div
            a, b = parent
            if a !== nothing
                a.grad += v.grad * (1.0 / b.value)
                push!(visited, a)
                backward(a, visited)
            end
            if b !== nothing
                b.grad += v.grad * (-a.value / (b.value^2))
                push!(visited, b)
                backward(b, visited)
            end
        end
    end
end


end