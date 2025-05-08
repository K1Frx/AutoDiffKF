include("../src/autodiffkf.jl")  # Załadowanie modułu Autodiffkf
using .Autodiffkf: Variable, +, *, -, /, exp, log, relu, sigmoid, tanh, backward  # Załadowanie funkcji z nowego modułu

function print_results(x::String, y::Vector{Variable})
    println("Wyniki testu: ", x)
    for i in 1:length(y)
        println("x: ", y[i].value, ", x_grad: ", y[i].grad)
    end
end

# Funkcja do testowania dodawania
function test_addition_two_variables()
    x = Variable(3.0)
    y = Variable(2.0)
    z = x + y
    backward(z)
    if z.value == 5.0 && x.grad == 1.0 && y.grad == 1.0
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_addition_three_variables()
    x = Variable(3.0)
    y = Variable(2.0)
    z = Variable(1.0)
    w = (x + y) + z
    backward(w)
    if w.value == 6.0 && x.grad == 1.0 && y.grad == 1.0 && z.grad == 1.0
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_multiplication()
    x = Variable(3.0)
    y = Variable(5.0)
    z = x * y
    backward(z)
    if z.value == 15.0 && x.grad == 5.0 && y.grad == 3.0
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_relu()
    x = Variable(-3.0)
    y = relu(x)
    backward(y)
    if y.value == 0.0 && x.grad == 0.0
        println("Sukces!")
    else
        println("Błąd!")
    end

    x = Variable(3.0)
    y = relu(x)
    backward(y)
    if y.value == 3.0 && x.grad == 1.0
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_sigmoid()
    x = Variable(0.0)
    y = sigmoid(x)
    backward(y)
    if y.value == 0.5 && x.grad == 0.25
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_tanh()
    x = Variable(0.5)
    y = tanh(x)
    backward(y)
    if y.value == tanh(0.5) && x.grad == 1 - tanh(0.5)^2
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_log()
    x = Variable(2.0)
    y = log(x)
    backward(y)
    if y.value == log(2.0) && x.grad == 0.5
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_exp()
    x = Variable(1.0)
    y = exp(x)
    backward(y)
    if y.value == exp(1.0) && x.grad == exp(1.0)
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function run_all_tests()
    test_addition_two_variables()
    test_addition_three_variables()
    test_multiplication()
    test_relu()
    test_sigmoid()
    test_tanh()
    test_log()
    test_exp()

    println("\nWszystkie testy zakończone.")
end

# Uruchomienie wszystkich testów
run_all_tests()