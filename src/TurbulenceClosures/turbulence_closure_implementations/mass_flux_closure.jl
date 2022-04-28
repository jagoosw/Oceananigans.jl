using Oceananigans.Architectures: architecture, device_event, arch_array
using Oceananigans.BuoyancyModels: top_buoyancy_flux, z_dot_g_b, buoyancy_perturbation
using Oceananigans.Operators
using KernelAbstractions.Extras.LoopInfo: @unroll

struct MassFluxVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    εg :: FT
    a₁ :: FT
    α  :: FT
    β₁ :: FT
    β₂ :: FT
    Cₘ :: FT 
    Cg :: FT
end

MassFluxVerticalDiffusivity{TD}(εg::FT, a₁::FT, α::FT, β₁::FT, β₂::FT, Cₘ::FT, Cg::FT) where {TD, FT} =
        MassFluxVerticalDiffusivity{TD, FT}(εg, a₁, α, β₁, β₂, Cₘ, Cg)

const MF = MassFluxVerticalDiffusivity

function MassFluxVerticalDiffusivity(time_discretization=ExplicitTimeDiscretization(), 
                                    FT = Float64;
                                    εg=0.1,
                                    a₁=0.667, 
                                    α=0.2, 
                                    β₁=0.9, 
                                    β₂=0.9, 
                                    Cₘ=0.065, 
                                    Cg=9.80665) 

    TD = typeof(time_discretization)

    return MassFluxVerticalDiffusivity{TD}(εg, a₁, α, β₁, β₂, Cₘ, Cg)
end

MassFluxVerticalDiffusivity(FT::DataType; kw...) =
    MassFluxVerticalDiffusivity(ExplicitTimeDiscretization(), FT; kw...)

# To change!
viscosity(::MF{TD, FT}, args...) where {TD, FT}   = zero(FT)
diffusivity(::MF{TD, FT}, args...) where {TD, FT} = zero(FT)

#####
##### Diffusivity field utilities
#####

# Support for "ManyIndependentColumnMode"
const MFArray = AbstractArray{<:MF}
const FlavorOfMF = Union{MF, MFArray}

with_tracers(tracers, closure::FlavorOfMF) = closure

"""
Mass Flux turbulence closure 

```math
⟨w′ψ′⟩ = aₚwₚ⋅(ψ - ψₚ) 
```

Closure equations:

vertical plume velocity
```math
(α + 0.5) (∂z wₚ²) + Cᵣ wₚ² = Bₚ
```

fractional plume area
```math
∂z aₚ = aₚ ⋅ ( wₚ⁻¹ ∂z wₚ + εₐₚ - δₐₚ )
```

plume properties
```math
∂z ψₚ + εₚ ψₚ = εₚ ψ
```
"""
# Mass flux parameters at the Faces (wₚ, aₚ) while (εₚ, ψₚ) at Centers 
@kernel function compute_plume_properties!(wₚ, aₚ, εₚ, ψₚ, grid, closure, buoyancy, tracers, tracer_bcs)
    i, j = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    
    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, tracers)
    w★ = abs(Qᵇ)^(1/3) 
    u★ = 0.0
    uₛ  = min(1.0, max(1e-5, 2/3*(w★ + u★)))
    aₚ₀ = ifelse(Qᵇ > 0, closure_ij.Cₘ * w★ / 2 * 3 / uₛ, 0.0)
    wₚ² = 0

    ## Calculating plume's vertical velocity
    wₚ[i, j, grid.Nz + 1] = 0
    @unroll for k in grid.Nz : -1 : 1
        bₚ    = buoyancy_perturbation(i, j, k, grid, buoyancy.model, ψₚ)
        bprod = buoyancy_production(i, j, k, grid, closure_ij, buoyancy, tracers, bₚ)
        cres  = cnvctive_resistance(i, j, k, grid, closure_ij, bₚ, εₚ)

        bprod *= Δzᶜᶜᶜ(i, j, k, grid)
        cres  *= Δzᶜᶜᶜ(i, j, k, grid)

        cₖ₊₁ = (closure_ij.α + 0.5) - cres / 2 
        cₖ   = (closure_ij.α + 0.5) + cres / 2 

        # updating wₚ²
        wₚ²  = max((bprod + wₚ² * cₖ₊₁) / cₖ, 0.0)
        wₚ[i, j, k] = sqrt( wₚ² )
    end
    
    ## Calculating plume's areas
    aₚ[i, j, grid.Nz+1] = min(aₚ₀, 0.1)
    @unroll for k in grid.Nz : -1 : 1

        wᵢ = max(wₚ[i, j, k+1], wₚ[i, j, k])
        if wᵢ > 0
            L = - ∂zᶜᶜᶜ(i, j, k, grid, wₚ) / wᵢ

            εₐₚ =   closure_ij.β₁ * max(0, L)
            δₐₚ = - closure_ij.β₂ * min(0, L)

            cₖ₊₁ = 1 + (L + εₐₚ - δₐₚ) / 2 * Δzᶜᶜᶜ(i, j, k, grid)
            cₖ   = 1 - (L + εₐₚ - δₐₚ) / 2 * Δzᶜᶜᶜ(i, j, k, grid)
            
            aₚ[i, j, k] = aₚ[i, j, k+1] * cₖ₊₁ / cₖ
        else
            aₚ[i, j, k] = 0.0
        end
        aₚ[i, j, k] = min(max(aₚ[i, j, k], 0.0), 0.1)
    end

    ## Calculating entrainment coefficient
    @unroll for k in grid.Nz : -1 : 1
        wᵢ = ℑzᵃᵃᶜ(i, j, k, grid, wₚ)
        εₚ[i, j, k] = wᵢ > 0 ? abs(∂zᶜᶜᶜ(i, j, k, grid, wₚ)) / wᵢ + closure_ij.εg : 0
    end

    ## Calculating plume's properties
    for (idx, ψ) in enumerate(ψₚ)
        tr = tracers[idx]
        @unroll for k in grid.Nz : -1 : 1
            L    = ℑzᵃᵃᶠ(i, j, k+1, grid, εₚ) * Δzᶜᶜᶠ(i, j, k, grid)
            cₖ₊₁ = 1 - 0.5 * L
            cₖ   = 1 + 0.5 * L
            ψ[i, j, k] = ifelse(wₚ[i, j, k+1] < 1e-12, 
                                tr[i, j, k],
                                (L * ℑzᵃᵃᶠ(i, j, k+1, grid, tr) + ψ[i, j, k+1] * cₖ₊₁) / cₖ) 
        end
    end
end

function calculate_diffusivities!(diffusivities, closure::FlavorOfMF, model)
    grid = model.grid
    arch = model.architecture
    tracers = model.tracers
    buoyancy = model.buoyancy

    # Start populating plume properties
    if model.clock.time == 0
        for (tracer_idx, tracer) in enumerate(model.tracers)
            diffusivities.ψₚ[tracer_idx] .= tracer
        end
    end

    tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    event = launch!(arch, grid, :xy,
                    compute_plume_properties!, 
                    diffusivities.wₚ,
                    diffusivities.aₚ,
                    diffusivities.εₚ,
                    diffusivities.ψₚ,
                    grid, closure, buoyancy, tracers, tracer_bcs,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

function DiffusivityFields(grid, tracer_names, user_bcs, ::FlavorOfMF)
    aₚ = ZFaceField(grid)
    wₚ = ZFaceField(grid)
    εₚ = CenterField(grid)

    ψₚ_tracers = []

    for i = 1:length(tracer_names)
        push!(ψₚ_tracers, CenterField(grid))
    end

    ψₚ = NamedTuple{tracer_names}(Tuple(ψₚ_tracers))

    return (; aₚ, wₚ, εₚ, ψₚ)
end

"""
Buoyancy production of plume's vertical velocity

```math
Bₚ = a₁ g (ρₚ - ρₑ) / ρₑ 
```
"""
@inline function buoyancy_production(i, j, k, grid, closure, buoyancy, tracers, bₚ) 
    bₑ = buoyancy_perturbation(i, j, k, grid, buoyancy.model, tracers)
    return closure.a₁ * closure.Cg * (bₚ - bₑ) / (bₑ + closure.Cg)
end

"""
Resistance to downwards convection

```math
Cᵣ = α g (ρₚ / pₕ) + εₚ
```
"""
@inline cnvctive_resistance(i, j, k, grid, closure, bₚ, εₚ) =  
        - closure.α * (bₚ + closure.Cg) / znode(Center(), Center(), Center(), i, j, k, grid) + εₚ[i, j, k]

####
#### Explicit time discretization fluxes
####

@inline c_minus_ψₚ(i, j, k, grid, C, K, ::Val{id}) where id = C[id][i, j, k] - K.ψₚ[id][i, j, k]

@inline diffusive_flux_z(i, j, k, grid, ::MF, K, ::Val{id}, U, C, clk, b) where id = 
       K.aₚ[i, j, k] * K.wₚ[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, c_minus_ψₚ, C, K, Val(id))

####
#### Shenanigans for implicit time discretization
####

