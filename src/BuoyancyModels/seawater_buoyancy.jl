using Oceananigans.BoundaryConditions: NoFluxBoundaryCondition

"""
    SeawaterBuoyancy{FT, EOS, T, S} <: AbstractBuoyancyModel{EOS}

BuoyancyModels model for seawater. `T` and `S` are either `nothing` if both
temperature and salinity are active, or of type `FT` if temperature
or salinity are constant, respectively.
"""
struct SeawaterBuoyancy{FT, EOS, T, S} <: AbstractBuoyancyModel{EOS}
             equation_of_state :: EOS
    gravitational_acceleration :: FT
          constant_temperature :: T
             constant_salinity :: S
end

required_tracers(::SeawaterBuoyancy) = (:T, :S)
required_tracers(::SeawaterBuoyancy{FT, EOS, <:Nothing, <:Number}) where {FT, EOS} = (:T,) # active temperature only
required_tracers(::SeawaterBuoyancy{FT, EOS, <:Number, <:Nothing}) where {FT, EOS} = (:S,) # active salinity only

"""
    SeawaterBuoyancy([FT=Float64;] gravitational_acceleration = g_Earth,
                                  equation_of_state = LinearEquationOfState(FT),
                                  constant_temperature = false, constant_salinity = false)

Returns parameters for a temperature- and salt-stratified seawater buoyancy model
with a `gravitational_acceleration` constant (typically called 'g'), and an
`equation_of_state` that related temperature and salinity (or conservative temperature
and absolute salinity) to density anomalies and buoyancy.

`constant_temperature` indicates that buoyancy depends only on salinity. For a nonlinear
equation of state, `constant_temperature` is used as the temperature of the system.
`true`. The same logic with the role of salinity and temperature reversed holds when `constant_salinity`
is provided.

For a linear equation of state, the values of `constant_temperature` or `constant_salinity` are irrelevant;
in this case, `constant_temperature=true` (and similar for `constant_salinity`) is valid input.
"""
function SeawaterBuoyancy(                        FT = Float64;
                          gravitational_acceleration = g_Earth,
                                   equation_of_state = LinearEquationOfState(FT),
                                constant_temperature = nothing,
                                   constant_salinity = nothing)

    # Input validation: convert constant_temperature or constant_salinity = true to zero(FT).
    # This method of specifying constant temperature or salinity in a SeawaterBuoyancy model
    # should only be used with a LinearEquationOfState where the constant value of either temperature
    # or sailnity is irrelevant.
    constant_temperature = constant_temperature === true ? zero(FT) : constant_temperature
    constant_salinity = constant_salinity === true ? zero(FT) : constant_salinity

    return SeawaterBuoyancy{FT, typeof(equation_of_state), typeof(constant_temperature), typeof(constant_salinity)}(
                            equation_of_state, gravitational_acceleration, constant_temperature, constant_salinity)
end

const TemperatureSeawaterBuoyancy = SeawaterBuoyancy{FT, EOS, <:Nothing, <:Number} where {FT, EOS}
const SalinitySeawaterBuoyancy = SeawaterBuoyancy{FT, EOS, <:Number, <:Nothing} where {FT, EOS}

@inline get_temperature_and_salinity(::SeawaterBuoyancy, C) = C.T, C.S
@inline get_temperature_and_salinity(b::TemperatureSeawaterBuoyancy, C) = C.T, b.constant_salinity
@inline get_temperature_and_salinity(b::SalinitySeawaterBuoyancy, C) = b.constant_temperature, C.S

@inline function buoyancy_perturbation(i, j, k, grid, b::SeawaterBuoyancy, C)
    θ, sᴬ = get_temperature_and_salinity(b, C)
    return - (b.gravitational_acceleration * ρ′(i, j, k, grid, b.equation_of_state, θ, sᴬ)
              / b.equation_of_state.reference_density)
end

#####
##### Buoyancy gradient components
#####

"""
    ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the x-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_x b = g ( α ∂_x θ - β ∂_x sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `θ` is
conservative temperature, and `sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂x_θ`, `∂x_sᴬ`, `α`, and `β` are all evaluated at cell interfaces in `x`
and cell centers in `y` and `z`.
"""
@inline function ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    θ, sᴬ = get_temperature_and_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, θ, sᴬ) * ∂xᶠᵃᵃ(i, j, k, grid, θ)
        - haline_contractionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, θ, sᴬ) * ∂xᶠᵃᵃ(i, j, k, grid, sᴬ) )
end

"""
    ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the y-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_y b = g ( α ∂_y θ - β ∂_y sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `θ` is
conservative temperature, and `sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂y_θ`, `∂y_sᴬ`, `α`, and `β` are all evaluated at cell interfaces in `y`
and cell centers in `x` and `z`.
"""
@inline function ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    θ, sᴬ = get_temperature_and_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, θ, sᴬ) * ∂yᵃᶠᵃ(i, j, k, grid, θ)
        - haline_contractionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, θ, sᴬ) * ∂yᵃᶠᵃ(i, j, k, grid, sᴬ) )
end

"""
    ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the vertical derivative of buoyancy for temperature and salt-stratified water,

```math
∂_z b = N^2 = g ( α ∂_z θ - β ∂_z sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `θ` is
conservative temperature, and `sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂z_θ`, `∂z_sᴬ`, `α`, and `β` are all evaluated at cell interfaces in `z`
and cell centers in `x` and `y`.
"""
@inline function ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    θ, sᴬ = get_temperature_and_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, θ, sᴬ) * ∂zᵃᵃᶠ(i, j, k, grid, θ)
        - haline_contractionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, θ, sᴬ) * ∂zᵃᵃᶠ(i, j, k, grid, sᴬ) )
end

#####
##### top buoyancy flux
#####

@inline get_temperature_and_salinity_flux(::SeawaterBuoyancy, bcs) = bcs.T, bcs.S
@inline get_temperature_and_salinity_flux(::TemperatureSeawaterBuoyancy, bcs) = bcs.T, NoFluxBoundaryCondition()
@inline get_temperature_and_salinity_flux(::SalinitySeawaterBuoyancy, bcs) = NoFluxBoundaryCondition(), bcs.S

@inline function top_bottom_buoyancy_flux(i, j, k, grid, b::SeawaterBuoyancy, top_bottom_tracer_bcs, clock, fields)
    θ, sᴬ = get_temperature_and_salinity(b, fields)
    θ_flux_bc, sᴬ_flux_bc = get_temperature_and_salinity_flux(b, top_bottom_tracer_bcs)

    θ_flux = getbc(θ_flux_bc, i, j, grid, clock, fields)
    sᴬ_flux = getbc(sᴬ_flux_bc, i, j, grid, clock, fields)

    return b.gravitational_acceleration * (
              thermal_expansionᶜᶜᶜ(i, j, k, grid, b.equation_of_state, θ, sᴬ) * θ_flux
           - haline_contractionᶜᶜᶜ(i, j, k, grid, b.equation_of_state, θ, sᴬ) * sᴬ_flux)
end

@inline    top_buoyancy_flux(i, j, grid, b::SeawaterBuoyancy, args...) = top_bottom_buoyancy_flux(i, j, grid.Nz, grid, b, args...)
@inline bottom_buoyancy_flux(i, j, grid, b::SeawaterBuoyancy, args...) = top_bottom_buoyancy_flux(i, j, 1, grid, b, args...)
    
