using Somoclu
using Test

import Somoclu: distance

# This is an Euclidean distance estimate computed from Julia,
function distance(p1::Ptr{Cfloat}, p2::Ptr{Cfloat}, d::Cuint)
    s = Cfloat(0.0)
    for i = 1:d
        v = unsafe_load(p1, i) - unsafe_load(p2, i)
        s += v*v
    end
    (isinf(s) || isnan(s)) && throw(ErrorException("Invalid distance computed."))
    return sqrt(s)::Cfloat
end

function deterministic_codebook(useCustomDistance=false, usePCA=false)
    ncolumns, nrows = 2, 2
    initialcodebook = zeros(Float32, 2, ncolumns*nrows)
    som = Som(ncolumns, nrows,
              initialization=usePCA ? "pca" : "random",
              initialcodebook=initialcodebook,
              useCustomDistance=useCustomDistance)
    println("useCustomDistance: $(som.useCustomDistance)")
    data = [0.1f0 0.2f0; 0.3f0 0.4f0]
    train!(som, data)
    correct_codebook = [0.15f0  0.126894f0 0.173106f0 0.15f0;
                        0.35f0  0.326894f0 0.373106f0 0.35f0]
    return sum(som.codebook - correct_codebook) < 10f-6
end

@test deterministic_codebook()
@test deterministic_codebook(true)
@test deterministic_codebook(false, true)
