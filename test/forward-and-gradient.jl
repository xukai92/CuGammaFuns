# TODO: use @test to do tests
import SpecialFunctions
using Flux: Tracker
using CuArrays: cu
using CuGammaFuns

xs_lgamma = [0.5f0]
xs_digamma = [2.0f0, -2.2f0]
xs_lbeta = ([2.0f0], [3.0f0])

println("1. Forward pass with CPU")

println(SpecialFunctions.lgamma.(xs_lgamma))
println(SpecialFunctions.digamma.(xs_digamma))
println(SpecialFunctions.lbeta.(xs_lbeta...))

println("2. Gradient with CPU")

lgamma_reduce_sum(x) = sum(SpecialFunctions.lgamma.(x))
digamma_reduce_sum(x) = sum(SpecialFunctions.digamma.(x))
lbeta_reduce_sum(x, y) = sum(SpecialFunctions.lbeta.(x, y))

println(Tracker.gradient(lgamma_reduce_sum, xs_lgamma))  
println(Tracker.gradient(digamma_reduce_sum, xs_digamma))  
println(Tracker.gradient(lbeta_reduce_sum, xs_lbeta...))  

println("3. Forward pass with GPU")

xs_lgamma_cu = cu(xs_lgamma)
xs_digamma_cu = cu(xs_digamma)
xs_lbeta_cu = map(x -> cu(x), xs_lbeta)

println(lgamma.(xs_lgamma_cu))
println(digamma.(xs_digamma_cu))
println(lbeta.(xs_lbeta_cu...))

println("4. Gradient with GPU")

lgamma_reduce_sum_cu(x) = sum(lgamma.(x))
digamma_reduce_sum_cu(x) = sum(digamma.(x))
lbeta_reduce_sum_cu(x, y) = sum(lbeta.(x, y))

println(Tracker.gradient(lgamma_reduce_sum_cu, xs_lgamma_cu))  
println(Tracker.gradient(digamma_reduce_sum_cu, xs_digamma_cu))
println(Tracker.gradient(lbeta_reduce_sum_cu, xs_lbeta_cu...))
