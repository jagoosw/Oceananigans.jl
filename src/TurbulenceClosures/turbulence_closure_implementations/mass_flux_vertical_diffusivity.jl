using Oceananigans.Architectures: architecture, device_event, arch_array
using Oceananigans.BuoyancyModels: top_buoyancy_flux, z_dot_g_b, buoyancy_perturbation
using Oceananigans.Operators
using KernelAbstractions.Extras.LoopInfo: @unroll

struct MassFluxVerticalDiffusivity{TD, E, A1, A2, RA, B1, B2, C} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    εg :: E
    a₁ :: A1
    α  :: A2
    Rᵅ :: RA
    β₁ :: B1
    β₂ :: B2
    Cₘ :: C 

    function MassFluxVerticalDiffusivity{TD}(εg::E, a₁::A1, α::A2, Rᵅ::RA, 
                                             β₁::B1, β₂::B2, Cₘ::C) where {TD, E, A1, A2, RA, B1, B2, C}
        return new{TD, E, A1, A2, RA, B1, B2, C}(εg, a₁, α, Rᵅ::RA, β₁, β₂, Cₘ)
    end
end

const MF = MassFluxVerticalDiffusivity

function MassFluxVerticalDiffusivity(; εg=0.001, a₁=1, α=0.2, Rᵅ=100, β₁=0.9, β₂=0.9, Cₘ=-0.065) where TD
    return MassFluxVerticalDiffusivity{ExplicitTimeDiscretization}(εg, a₁, α, Rᵅ, β₁, β₂, Cₘ)
end

#####
##### Diffusivity field utilities
#####

# Support for "ManyIndependentColumnMode"
const MFArray = AbstractArray{<:MF}
const FlavorOfMF = Union{MF, MFArray}

with_tracers(tracers, closure::FlavorOfMF) = closure
@inline viscosity_location(::FlavorOfMF) = (Center(), Center(), Face())
@inline diffusivity_location(::FlavorOfMF) = (Center(), Center(), Face())

function calculate_diffusivities!(diffusivities, closure::FlavorOfMF, model)
    grid = model.grid
    arch = model.architecture
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    pressure = model.pressure.pHY′

    if (model.clock.time == 0) 
        for (tracer_index, ψₚ) in enumerate(diffusivities.ψₚ)
            ψ = tracers[tracer_index]
            ψₚ .= ψ
        end
    end

    event = launch!(arch, grid, :xyz,
                    compute_plume_velocity_rhs!, diffusivities, grid, closure, buoyancy, tracers,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    solve!(diffusivities.wₚ, diffusivities.wₚ_solver, diffusivities.wₚ_rhs, closure, buoyancy, pressure, diffusivities;
                    dependencies = device_event(arch))

    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    event = launch!(arch, grid, :xyz,
                    compute_plume_velocity!, diffusivities,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    event = launch!(arch, grid, :xy,
                    compute_plume_areas!, diffusivities, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    u_event = launch!(arch, grid, :xy,
                      compute_plume_properties!, diffusivities.uₚ.u, velocities.u, diffusivities.wₚ, grid, closure,
                      dependencies = device_event(arch))

    v_event = launch!(arch, grid, :xy,
                      compute_plume_properties!, diffusivities.uₚ.v, velocities.v, diffusivities.wₚ, grid, closure,
                      dependencies = device_event(arch))

    events = [u_event, v_event]

    for (tracer_index, ψₚ) in enumerate(diffusivities.ψₚ)
        ψ = tracers[tracer_index]
        event = launch!(arch, grid, :xy,
                        compute_plume_properties!, ψₚ, ψ, diffusivities.wₚ, grid, closure,
                        dependencies = device_event(arch))

        push!(events, event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

@inline mf_upper_diagonal(i, j, k, grid, args...) =   1 / (2 * Δzᶜᶜᶠ(i, j, k, grid))
@inline mf_lower_diagonal(i, j, k, grid, args...) = - 1 / (2 * Δzᶜᶜᶠ(i, j, k, grid))

@inline function mf_diagonal(i, j, k, grid, closure, buoyancy, pressure, diffusivities) 
    plume_buoyancy = buoyancy_perturbation(i, j, k, grid, buoyancy.model, diffusivities.ψₚ)
    @show plume_buoyancy
    return ifelse(pressure[i, j, k] == 0, 1.0, closure.Rᵅ * (plume_buoyancy + 1) / pressure[i, j, k])
end

@kernel function compute_plume_velocity_rhs!(diffusivities, grid, closure, buoyancy, tracers)
    i, j, k = @index(Global, NTuple)
    diffusivities.wₚ_rhs[i, j, k] = closure.a₁ * z_dot_g_b(i, j, k, grid, buoyancy, tracers)
end

@kernel function compute_plume_velocity!(diffusivities)
    i, j, k = @index(Global, NTuple)
    diffusivities.wₚ[i, j, k] = - sqrt(abs(diffusivities.wₚ[i, j, k]))
end

@kernel function compute_plume_areas!(diffusivities, grid, closure, velocities, tracers, buoyancy, tracer_bcs)
    i, j = @index(Global, NTuple)

    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, merge(velocities, tracers))
    w★ = abs(Qᵇ)^(1/3) 
    u★ = 0.0

    if Qᵇ > 0
        diffusivities.aₚ[i, j, grid.Nz] = - closure.Cₘ * w★ / 2 * 3 / (w★ + u★)
    else
        diffusivities.aₚ[i, j, grid.Nz] = 0.0
    end

    @unroll for k in grid.Nz-1 : -1 : 1
        if !(is_stableᶜᶜᶠ(i, j, k, grid, tracers, buoyancy))
            L   = ∂zᶜᶜᶠ(i, j, k, grid, diffusivities.wₚ) / (ℑzᵃᵃᶜ(i, j, k, grid, diffusivities.wₚ) + 1e-16)
            εₐₚ =   closure.β₁ * max(0, L)
            δₐₚ = - closure.β₂ * min(0, L)
            diffusivities.aₚ[i, j, k] = diffusivities.aₚ[i, j, k+1] / exp(Δzᶜᶜᶠ(i, j, k, grid) * (- L + εₐₚ - δₐₚ))
        else
            diffusivities.aₚ[i, j, k] = 0
        end
    end
end

@kernel function compute_plume_properties!(ψₚ, ψ, wₚ, grid, closure)
    i, j = @index(Global, NTuple)

    ψₚ[i, j, grid.Nz+1] = ψ[i, j, grid.Nz+1]
    @unroll for k = grid.Nz:-1:1
        if wₚ[i, j, k] == 0
            ψₚ[i, j, k] = ψₚ[i, j, k+1]
        else
            rhs = abs(∂zᶜᶜᶠ(i, j, k, grid, wₚ) / (ℑzᵃᵃᶜ(i, j, k, grid, wₚ) + 1e-16)) + closure.εg
            ψₚ[i, j, k] = ψₚ[i, j, k+1] - Δzᶜᶜᶠ(i, j, k, grid) * rhs
        end
    end
end

function DiffusivityFields(grid, tracer_names, user_bcs, ::MF)
    aₚ = CenterField(grid)
    wₚ = CenterField(grid)


    wₚ_rhs = CenterField(grid)

    uₚ = Field((Face, Center, Center), grid)
    vₚ = Field((Center, Face, Center), grid)
    
    ψₚ_tracers = []

    for i = 1:length(tracer_names)
        push!(ψₚ_tracers, CenterField(grid))
    end

    ψₚ = NamedTuple{tracer_names}(Tuple(ψₚ_tracers))
    uₚ = (u = uₚ, v = vₚ)

    wₚ_solver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = mf_lower_diagonal,
                                        diagonal = mf_diagonal,
                                        upper_diagonal = mf_upper_diagonal)

    return (; aₚ, wₚ, uₚ, ψₚ, wₚ_solver, wₚ_rhs)
end

@inline a_times_w(i, j, k, grid, a, w) = a[i, j, k] * w[i, j, k]

####
#### Explicit time discretization fluxes
####

@inline viscous_flux_uz(i, j, k, grid,  ::MF, K, U, C, clock, b) = - ℑxᶠᵃᵃ(i, j, k, grid, a_times_w, K.aₚ, K.wₚ) * (U.u[i, j, k] - K.uₚ.u[i, j, k])
@inline viscous_flux_vz(i, j, k, grid,  ::MF, K, U, C, clock, b) = - ℑyᵃᶠᵃ(i, j, k, grid, a_times_w, K.aₚ, K.wₚ) * (U.v[i, j, k] - K.uₚ.v[i, j, k])

@inline diffusive_flux_z(i, j, k, grid, ::MF, K, ::Val{id}, U, C, clk, b) where id = - K.aₚ[i, j, k] * K.wₚ[i, j, k] * (C[id][i, j, k] - K.ψₚ[id][i, j, k])

####
#### Shenanigans for implicit time discretization
####

