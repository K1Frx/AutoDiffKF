include("../src/Autodiffkf-old.jl")  # Załadowanie modułu Autodiff
using .Autodiff: Variable, ad_sin, ad_add, backward, ad_mul # Załadowanie modułu Autodiff

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
    z = ad_add(x, y)
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
    w = ad_add(ad_add(x, y), z)
    backward(w)
    if w.value == 6.0 && x.grad == 1.0 && y.grad == 1.0 && z.grad == 1.0
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_sin()
    x = Variable(10.0)
    y = ad_sin(x)
    backward(y)

    if y.value == sin(10.0) && x.grad == cos(10.0)
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_sin_and_addition()
    x = Variable(5.0)
    y = Variable(3.0)
    z = ad_sin(ad_add(x, y))
    backward(z)
    if z.value == sin(5 + 3) && x.grad == cos(5 + 3) && y.grad == cos(5 + 3)
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_mul()
    x = Variable(3.0)
    y = Variable(5.0)
    z = ad_mul(x, y)
    backward(z)
    if z.value == 15.0 && x.grad == 5.0 && y.grad == 3.0
        println("Sukces!")
    else
        println("Błąd!")
    end
end

function test_advanced_mul()
    x1 = Variable(3.0)
    x2 = Variable(5.0)
    x3 = Variable(2.0)
    x4 = Variable(4.0)
    x5 = Variable(6.0)

    y = ad_mul(x5,ad_mul(ad_mul(x1, x2), ad_mul(x3, x4)))

    backward(y)
    if y.value == 720 && x1.grad == 240 && x2.grad == 144 && x3.grad == 360 && x4.grad == 180 && x5.grad == 120
        println("Sukces!")
    else
        println("Błąd!")
    end
end


function run_all_tests()
    test_addition_two_variables()
    test_addition_three_variables()
    test_sin()
    test_sin_and_addition()
    test_mul()
    test_advanced_mul()

    println("\nWszystkie testy zakończone.")
end

# Uruchomienie wszystkich testów
run_all_tests()