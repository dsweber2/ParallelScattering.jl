module ParallelScattering
using Distributed, SharedArrays
using LinearAlgebra, Interpolations, Wavelets, FFTW
using ContinuousWavelets
using SpecialFunctions, LinearAlgebra
using HDF5, JLD
using Plots

using ScatteringTransform
import ScatteringTransform: Scattered, ScatteredFull, ScatteredOut, depth, eltypes, scatteringTransform, makeTuple, processArgs
export scatteringTransform, stFlux, stParallel, eltypes, depth, ScatteredOut, ScatteredFull

# extra constructor methods for ScatteredFull and ScatteredOut
function ScatteredFull(layers::scatteringTransform{S,1}, X::Array{T,N}; totalScales = [-1 for i = 1:depth(layers)+1], outputSubsample = (-1, -1)) where {T<:Real,N,S}
    if N == 1
        X = reshape(X, (size(X)..., 1))
    end

    n, q, dataSizes, outputSizes, resultingSize =
        calculateSizes(layers, outputSubsample, size(X),
            totalScales = totalScales)
    numInLayer = getQ(layers, n, totalScales; product = false)
    addedLayers = [numInLayer[min(i, 2):i] for i = 1:depth(layers)+1]
    if 1 == N
        zerr = [zeros(T, n[i], addedLayers[i]...) for i = 1:depth(layers)+1]
        output = [zeros(T, resultingSize[i], addedLayers[i]...) for i = 1:depth(layers)+1]
    else
        zerr = [zeros(T, n[i], q[i], size(X)[2:end]...) for i = 1:depth(layers)+1]
        output = [zeros(T, resultingSize[i], addedLayers[i]..., size(X)[2:end]...)
                  for i = 1:depth(layers)+1]
        @info "" [size(x) for x in output]
    end
    zerr[1][:, 1, axes(X)[2:end]...] = copy(X)
    return ScatteredFull{T,N + 1}(depth(layers), 1, zerr, output)
end

function ScatteredOut(layers::ST, X::Array{T,N};
    totalScales = [-1 for i = 1:depth(layers)+1],
    outputSubsample = (-1, -1)) where {ST<:scatteringTransform,T<:Real,N,S}
    if N == 1
        X = reshape(X, (size(X)..., 1))
    end

    n, q, dataSizes, outputSizes, resultingSize =
        calculateSizes(layers, outputSubsample, size(X), totalScales = totalScales)
    addedLayers = getListOfScaleDims(layers, n, totalScales)
    @info addedLayers
    if 1 == N
        output = [zeros(T, resultingSize[i], addedLayers[i]...) for i = 1:depth(layers)+1]
    else
        output = [zeros(T, resultingSize[i], addedLayers[i]..., size(X)[2:end]...)
                  for i = 1:depth(layers)+1]
    end
    @info [size(x) for x in output]
    return ScatteredOut{T,N + 1}(depth(layers), 1, output)
end

@doc """
        st(X::Array{T, N}, layers::stParallel, nonlinear::nl; fullOr::fullType=fullType(),# subsam::Sub = bspline(), thin::Bool=true, outputSubsample::Tuple{Int, Int}=(-1,-1), subsam::Bool=true, totalScales = [-1 for i=1:depth(layers)+1], percentage = .9, fftPlans = -1) where {T <: Real, S <: Union, N, nl <: Function, Sub <: resamplingMethod}
        stParallel(m, Xlength; CWTType=morl, subsampling = 2, outputSize=[1,1], varargs...) -> layers
1D scattering transform using the stParallel layers. you can switch out the nonlinearity as well as the method of subsampling. Finally, the stType is a string. If it is "full", it will produce all paths. If it is "decreasing", it will only keep paths of increasing scale. If it is "collating", you must also include a vector of matrices.
# Arguments
- `nonlinear` : a type of nonlinearity. Should be a function that acts on Complex numbers
- `thin` : determines whether to wrap the output into a format that can be indexed using paths. `thin` cannot.
- `totalScales`, if positive, gives the number of non-averaging wavelets.
- `outputSubsample` is a tuple, with the first number indicating a rate, and the second number giving the minimum allowed size. If some of the entries are less than 1, it has different behaviour:
    + `(<1, x)` : subsample to x elements for each path.
    + `(<1, <1)` : no ssubsampling
    + `(x, <1)` : subsample at a rate of x, with at least one element kept in each path
- `fullOr::fullType=fullType()` : the structure of the transform either
       `fullType()`, `collatingType()` or `decreasingType()`. At the moment,
       only `fullType()` is functional.
- `fftPlans = false` if not `false`, it should be a 2D array of `Future`s, where the first index is the layer, and the second index the core. See `createFFTPlans` if you want to do this.
"""
struct stParallel{T,Dimension,Depth,subsampType,outType} <: scatteringTransform{Dimension,Depth}
    n::Tuple{Vararg{Int,Dimension}} # the dimensions of a single entry
    shears::Array{T,1} # the array of the transforms; the final of these is
    # used only for averaging, so it has length m+1
    subsampling::subsampType # for each layer, the rate of
    # subsampling. There is one of these for layer zero as well, since the
    # output is subsampled, so it should have length m+1
    outputSize::outType # a list of the size of a single output example
    # dimensions in each layer. The first index is layer, the second is
    # dimension (e.g. a 3 layer shattering transform is 3×2) TODO: currently
    # unused for the 1D case
end
function Base.show(io::IO, l::stParallel{T,D,Depth}) where {T,D,Depth}
    print(io, "stParallel{$T,$D} depth $(Depth), input size $(l.n), subsampling rates $(l.subsampling), outputsizes = $(l.outputSize)")
end
function eltypes(f::stParallel)
    return (eltype(f.shears), length(f.n))
end
# the fully specified version
# TODO: use a layered transform parameter to determine if we're returning a st or thinst instead
@doc """
 """
function stParallel(m, Xlength; CWTType = morl, subsampling = 2, outputSize = [1, 1], varargs...)
    CWTType = makeTuple(m + 1, CWTType)
    subsampling = makeTuple(m + 1, subsampling)
    pairedArgs = processArgs(m + 1, varargs)
    shears = map(x -> ContinuousWavelets.wavelet(x[1]; x[2]...), zip(CWTType, pairedArgs))
    stParallel{typeof(shears[1]),1,m,typeof(subsampling),typeof(outputSize)}((Xlength,), shears, subsampling, outputSize)
end

include("subsampling.jl")
export sizes, bsplineType, bilinearType
include("modifiedTransforms.jl")
include("Utils.jl")
export calcuateSizes, calculateThinStSizes, createFFTPlans, remoteMultiply,
    createRemoteFFTPlan, computeAllWavelets, plotAllWavelets, getQ
include("nonlinearities.jl")
export spInverse, aTanh, Tanh, ReLU, piecewiseLinear, plInverse
include("transform.jl")
export st, transformMidLayer!, transformFinalLayer!
include("pathMethods.jl")
export pathToThinIndex, plotCoordinate, numberSkipped, logabs, maxPooling, numScales,
    incrementKeeper, numInLayer
include("postProcessing.jl")
export MatrixAggrigator, reshapeFlattened, loadSyntheticMatFile, transformFolder, flatten, rollSt
end
