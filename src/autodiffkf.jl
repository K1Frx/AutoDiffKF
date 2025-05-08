module Autodiffkf
import Base: +, -, *, /, exp, log, tanh, min, max

export Variable, ReadableVariable, +, -, *, /, min, max, exp, log, relu, sigmoid, tanh, softmax
export zero_grad!, backward, print_gradients

mutable struct Variable
    value::Float64
    grad::Float64
    parents::Vector{Any}
    op_type::Symbol
    m::Float64
    v::Float64
end

function Variable(value::Float64)
    return Variable(value, 0.0, [], :none, 0.0, 0.0)
end

function Variable(value::Float32)
    return Variable(Float64(value), 0.0, [], :none, 0.0, 0.0)
end

function Variable(value::Int)
    return Variable(Float64(value), 0.0, [], :none, 0.0, 0.0)
end

function Variable(value::Float64, symbol::Symbol)
    return Variable(value, 0.0, [], symbol, 0.0, 0.0)
end

function ReadableVariable(x::Variable)
    return x.value
end

function +(a::Variable, b::Variable)
    result = Variable(a.value + b.value, :add)
    push!(result.parents, (a, b))
    return result
end

function -(a::Variable, b::Variable)
    result = Variable(a.value - b.value, :sub)
    push!(result.parents, (a, b))
    return result
end

function *(a::Variable, b::Variable)
    result = Variable(a.value * b.value, :mul)
    push!(result.parents, (a, b))
    return result
end

function /(a::Variable, b::Variable)
    result = Variable(a.value / b.value, :div)
    push!(result.parents, (a, b))
    return result
end

function exp(a::Variable)
    result = Variable(exp(a.value), :exp)
    push!(result.parents, a)
    return result
end

function log(a::Variable)
    result = Variable(log(a.value), :log)
    push!(result.parents, a)
    return result
end

function relu(a::Variable)
    result = Variable(max(0.0, a.value), :relu)
    push!(result.parents, a)
    return result
end

function sigmoid(a::Variable)
    value = 1.0 / (1.0 + exp(-a.value))
    result = Variable(value, :sigmoid)
    push!(result.parents, a)
    return result
end

function tanh(a::Variable)
    result = Variable(tanh(a.value), :tanh)
    push!(result.parents, a)
    return result
end

function softmax(a::Vector{Variable})
    max_val = maximum([x.value for x in a])
    exp_vals = [exp(x - max_val) for x in a]
    sum_exp_vals = sum(exp_vals)
    result = [x / sum_exp_vals for x in exp_vals]
    return result
end

function zero_grad!(a::Variable)
    a.grad = 0.0
end

function zero_grad!(a::Vector{Variable})
    for v in a
        zero_grad!(v)
    end
end

function min(a::Variable, b::Variable)
    result = Variable(min(a.value, b.value), :min)
    push!(result.parents, (a, b))
    return result
end

function max(a::Variable, b::Variable)
    result = Variable(max(a.value, b.value), :max)
    push!(result.parents, (a, b))
    return result
end

function zero_grad!(a::Matrix{Variable})
    for row in eachrow(a)
        for element in row
            zero_grad!(element)
        end
    end
end

function print_gradients(v::Variable, visited::Set{Variable}=Set{Variable}())
    if v in visited && all(parent in visited for parent in v.parents)
        return
    end

    push!(visited, v)

    println("Variable: $(v.value), Gradient: $(v.grad), Op Type: $(v.op_type)")

    for parent in v.parents
        if parent isa Variable
            print_gradients(parent, visited)
        else
            a, b = parent 
            print_gradients(a, visited)
            print_gradients(b, visited)
        end
    end
end

function backward(v::Variable, visited::Set{Variable}=Set{Variable}())
    if v in visited
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
                backward(a, visited)
            end
            if b !== nothing
                b.grad += v.grad * 1
                backward(b, visited)
            end

        elseif v.op_type == :sub
            a, b = parent
            if a !== nothing
                a.grad += v.grad * 1
                backward(a, visited)
            end
            if b !== nothing
                b.grad += v.grad * -1
                backward(b, visited)
            end

        elseif v.op_type == :mul
            a, b = parent
            if a !== nothing
                a.grad += v.grad * b.value
                backward(a, visited)
            end
            if b !== nothing
                b.grad += v.grad * a.value
                backward(b, visited)
            end

        elseif v.op_type == :log
            a = parent
            if a !== nothing
                a.grad += v.grad * (1 / a.value)
                backward(a, visited)
            end

        elseif v.op_type == :exp
            a = parent
            if a !== nothing
                a.grad += v.grad * exp(a.value)
                backward(a, visited)
            end

        elseif v.op_type == :relu
            a = parent
            if a !== nothing
                a.grad += v.grad * (a.value > 0.0 ? 1.0 : 0.0)
                backward(a, visited)
            end

        elseif v.op_type == :sigmoid
            a = parent
            if a !== nothing
                a.grad += v.grad * (v.value * (1.0 - v.value))
                backward(a, visited)
            end

        elseif v.op_type == :tanh
            a = parent
            if a !== nothing
                a.grad += v.grad * (1 - tanh(a.value)^2)
                backward(a, visited)
            end

        elseif v.op_type == :div
            a, b = parent
            if a !== nothing
                a.grad += v.grad * (1.0 / b.value)
                backward(a, visited)
            end
            if b !== nothing
                b.grad += v.grad * (-a.value / (b.value^2))
                backward(b, visited)
            end
        
        elseif v.op_type == :min
            a, b = parent
            if a !== nothing && a.value <= b.value
                a.grad += v.grad
                backward(a, visited)
            end
            if b !== nothing && b.value < a.value
                b.grad += v.grad
                backward(b, visited)
            end
        
        elseif v.op_type == :max
            a, b = parent
            if a !== nothing && a.value >= b.value
                a.grad += v.grad
                backward(a, visited)
            end
            if b !== nothing && b.value > a.value
                b.grad += v.grad
                backward(b, visited)
            end
        end
    end
end

end