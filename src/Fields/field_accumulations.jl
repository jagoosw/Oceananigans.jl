#####
##### Reductions of AbstractField
#####

struct Accumulation{D, R, O, M}
    operator :: R
    operand :: O
    dims :: M
    direction :: D
end

struct AccumulateFowards end
struct AccumulateBackwards end

"""
    Accumulation(operator, operand; dims)
"""
Accumulation(op, operand; dims) = Accumulation(op, operand, dims)

location(a::Accumulation) = location(a.operand)

function Field(accumulation::Accumulation;
               data = nothing,
               indices = indices(accumulation.operand),
               recompute_safely = false)

    operand = accumulation.operand
    grid = operand.grid
    LX, LY, LZ = loc = location(accumulation)
    indices = reduced_indices(indices; dims=reduction.dims)

    if isnothing(data)
        data = new_data(grid, loc, indices)
        recompute_safely = false
    end

    boundary_conditions = FieldBoundaryConditions(grid, loc, indices)
    status = recompute_safely ? nothing : FieldStatus()

    return Field(loc, grid, data, boundary_conditions, indices, accumulation, status)
end

const AccumulatedComputedField{D} = Field{<:Any, <:Any, <:Any, <:Accumulation{D}} where D

# "Forward" accumulation from 1 to N
function compute!(field::AccumulatedComputedField, time=nothing)
    accumulation = field.operand
    compute_at!(accumulation.operand, time)
    accumulate!(accumulation.operator, field, accumulation.operand)
    return field
end

# "Backward" accumulation from N to 1
# Support 3 cases: dims=1, 2, 3

function compute!(field::AccumulatedComputedField{<:AccumulateBackwards}, time=nothing)
    accumulation = field.operand
    grid = field.grid
    compute_at!(accumulation.operand, time)

    # Example
    # dims = 3
    backwards_accumulate = backwards_accumulate_z! 
    Nfirst = grid.Nz
    Nlast = 1
    layout = :xy

    arch = architecture(comp)
    event = launch!(arch, field.grid, layout,
                    backwards_accumulate!, field.data, accumulation.operand, Nfirst, Nlast)
    wait(device(arch), event)

    return field
end

@kernel function backwards_accumulate_x!(data, accumulation, Nfirst, Nlast)
    j, k = @index(Global, NTuple)
    @inbounds data[Nfirst, j, k] = accumulation.operand[Nfirst, j, k]
    @unroll for i in Nfirst - 1 : -1 : Nlast
        @inbounds data[i, j, k] = data[i+1, j, k] + accumulation.operand[i, j, k]
    end
end

@kernel function backwards_accumulate_y!(data, accumulation, Nfirst, Nlast)
    i, k = @index(Global, NTuple)
    @inbounds data[i, Nfirst, k] = accumulation.operand[i, Nfirst, k]
    @unroll for j in Nfirst - 1 : -1 : Nlast
        @inbounds data[i, j, k] = data[i, j+1, k] + accumulation.operand[i, j, k]
    end
end

@kernel function backwards_accumulate_z!(data, accumulation, Nfirst, Nlast)
    i, j = @index(Global, NTuple)
    @inbounds data[i, j, Nfirst] = accumulation.operand[i, j, Nfirst]
    @unroll for k in Nfirst - 1 : -1 : Nlast
        @inbounds data[i, j, k] = data[i, j, k+1] + accumulation.operand[i, j, k]
    end
end

#####
##### show
#####

Base.show(io::IO, field::AccumulatedComputedField) =
    print(io, "$(summary(field))\n",
          "├── data: $(typeof(field.data)), size: $(size(field))\n",
          "├── grid: $(summary(field.grid))\n",
          "├── operand: $(summary(field.operand))\n",
          "└── status: $(summary(field.status))")

#=
Base.summary(r::Accumulation) = string(r.reduce!, 
                                    " over dims ", r.dims,
                                    " of ", summary(r.operand))

Base.show(io::IO, r::Reduction) =
    print(io, "$(summary(r))\n",
          "└── operand: $(summary(r.operand))\n",
          "    └── grid: $(summary(r.operand.grid))")
=#
