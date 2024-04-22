using Lux, Optimisers, Random, Statistics, Zygote
using CairoMakie
using PyCall, NPZ
using JLD2

np = pyimport("numpy")

nk = 100
nz = 20
nc = 2000
ncv = 1000

cicli = 20
epoche = 20000
div = 5

input = npzread("../files/train_in_emul_" * string(nc) * ".npy")
output = npzread("../files/train_pkz_emul_" * string(nc) * ".npy")

x = input
y = np.reshape(output, [nc,nk*nz])'

# standardizzazione del trainig dataset (feature - min(features))/(max(features) - min(features))
min_in = minimum(x, dims=2)
max_in = maximum(x, dims=2)

min_out = minimum(y, dims=2)
max_out = maximum(y, dims=2)

x = (x .- min_in) ./ (max_in .- min_in)
y = (y .- min_out) ./ (max_out .- min_out)


zz = Float32.(range(0, 5, 20))
h = 0.67810
kk = exp10.(LinRange(-4, log10(3), 100)) .* h

println("x: ", size(x), '\t', "y: ", size(y))


# creo architettura della rete
layer_size = 64
in_size = size(x)[1]
out_size = size(y)[1]

nn = Chain(
    Dense(in_size => layer_size, tanh),
    Dense(layer_size => layer_size, tanh),
    Dense(layer_size => layer_size, tanh),
    Dense(layer_size => layer_size, tanh),
    Dense(layer_size => layer_size, tanh),
    Dense(layer_size => out_size)
)

# ottimizzatore
learning_rate = 0.001f0
opt = Adam(learning_rate)

# imposto lo stato iniziale della rete con parametri estratti casualmente
rng = MersenneTwister()
Random.seed!(rng, 12345)
tstate = Lux.Training.TrainState(rng, nn, opt)

# regola per calcolare i gradienti
vjp_rule = Lux.Training.AutoZygote()

# definisco la loss function
function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = mean(abs2, y_pred .- data[2])
    return mse_loss, st, ()
end

# importo il validation dataset e calcolo la loss sul validation
valid_in = npzread("../files/val_in_emul_" * string(ncv) * ".npy")
valid_in = (valid_in .- min_in) ./ (max_in .- min_in)
valid_out = npzread("../files/val_pkz_emul_" * string(ncv) * ".npy")
valid_out = np.reshape(valid_out, [ncv,nk*nz])'
valid_out = (valid_out .- min_out) ./ (max_out .- min_out)

val_data = (valid_in, valid_out)


# alleno la rete
initial_loss_validation = loss_function(nn, tstate.parameters, tstate.states, val_data)[1]
function main(tstate::Lux.Experimental.TrainState, vjp, data, epochs, i)

    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(
            vjp, loss_function, data, tstate)
        tstate = Lux.Training.apply_gradients(tstate, grads)

        loss_validation = loss_function(nn, tstate.parameters, tstate.states, val_data)[1]
        if loss_validation < initial_loss_validation
            jldsave("../models/nn_$(cicli)-$(epoche)_$(div).jld2"; tstate)
            global initial_loss_validation = loss_validation
            println("Epoch: $(i).$(epoch)\t|| Training Loss: $(loss)\t|| Validation Loss: $loss_validation")
        else 
            println("Epoch: $(i).$(epoch)\t|| Training Loss: $(loss)")
        end
    end
    return tstate
end

dev_cpu = cpu_device()

for i in 1:cicli
    global tstate = main(tstate, vjp_rule, (x, y), epoche, i)
    global learning_rate = learning_rate/div
    global opt = Adam(learning_rate)
end

# salvo lo stato allenato della rete
# jldsave("../models/nn_$(cicli)-$(epoche)_$(div).jld2"; trained_state)