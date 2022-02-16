# tests for the various forms of stParallel for the ScatteringTransform
# exit()
# using Revise
using Distributed
addprocs(min((Sys.CPU_THREADS) - 2 - nprocs(), 2))
@everywhere using Interpolations, ContinuousWavelets, Wavelets
@everywhere using FFTW
@everywhere using ParallelScattering
@everywhere using SharedArrays
using Test
using AbstractFFTs, LinearAlgebra, Statistics, Random

include("planTests.jl") # all from Parallel
include("cwtTests.jl") # also from Parallel
