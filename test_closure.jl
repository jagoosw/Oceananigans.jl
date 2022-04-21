using Oceananigans
using DataDeps
using JLD2

lesbrary_url = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/glw/lesbrary2/LESbrary/idealized/"

cases = ["free_convection",
         "weak_wind_strong_cooling",
         "strong_wind_weak_cooling",
         "strong_wind",
         "strong_wind_no_rotation"]

two_day_suite_url = lesbrary_url * "two_day_suite/"

glom_url(suite, resolution, case) = string(lesbrary_url,
                                           suite, "/", resolution, "_resolution/",
                                           case, "_instantaneous_statistics.jld2")

two_day_suite_1m_paths  = [glom_url( "two_day_suite", "2m_2m_1m", case) for case in cases]
two_day_suite_2m_paths  = [glom_url( "two_day_suite", "4m_4m_2m", case) for case in cases]
two_day_suite_4m_paths  = [glom_url( "two_day_suite", "8m_8m_4m", case) for case in cases]
four_day_suite_1m_paths = [glom_url("four_day_suite", "2m_2m_1m", case) for case in cases]
four_day_suite_2m_paths = [glom_url("four_day_suite", "4m_4m_2m", case) for case in cases]
four_day_suite_4m_paths = [glom_url("four_day_suite", "8m_8m_4m", case) for case in cases]
six_day_suite_1m_paths  = [glom_url( "six_day_suite", "2m_2m_1m", case) for case in cases]
six_day_suite_2m_paths  = [glom_url( "six_day_suite", "4m_4m_2m", case) for case in cases]
six_day_suite_4m_paths  = [glom_url( "six_day_suite", "8m_8m_4m", case) for case in cases]

dep = DataDep("two_day_suite_1m", "Idealized 2 day simulation data with 1m vertical resolution", two_day_suite_1m_paths)
DataDeps.register(dep)
dep = DataDep("two_day_suite_2m", "Idealized 2 day simulation data with 2m vertical resolution", two_day_suite_2m_paths)
DataDeps.register(dep)
dep = DataDep("two_day_suite_4m", "Idealized 2 day simulation data with 4m vertical resolution", two_day_suite_4m_paths)
DataDeps.register(dep)
dep = DataDep("four_day_suite_1m", "Idealized 4 day simulation data with 1m vertical resolution", four_day_suite_1m_paths)
DataDeps.register(dep)
dep = DataDep("four_day_suite_2m", "Idealized 4 day simulation data with 2m vertical resolution", four_day_suite_2m_paths)
DataDeps.register(dep)
dep = DataDep("four_day_suite_4m", "Idealized 4 day simulation data with 4m vertical resolution", four_day_suite_4m_paths)
DataDeps.register(dep)
dep = DataDep("six_day_suite_1m", "Idealized 6 day simulation data with 1m vertical resolution", six_day_suite_1m_paths)
DataDeps.register(dep)
dep = DataDep("six_day_suite_2m", "Idealized 6 day simulation data with 2m vertical resolution", six_day_suite_2m_paths)
DataDeps.register(dep)
dep = DataDep("six_day_suite_4m", "Idealized 6 day simulation data with 4m vertical resolution", six_day_suite_4m_paths)
DataDeps.register(dep)

cases = ["free_convection",
         "strong_wind_weak_cooling",
         "weak_wind_strong_cooling",
         "strong_wind",
         "strong_wind_no_rotation"]

datapaths = [@datadep_str("two_day_suite_1m/$(case)_instantaneous_statistics.jld2") for case in cases]

file = jldopen(datapaths[1])

top_T_flux = file["parameters/boundary_condition_θ_top"]
bot_T_flux = file["parameters/boundary_condition_θ_bottom"]

buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(thermal_expansion = 2e-4), constant_salinity = 35)

Nz = 16

grid = RectilinearGrid(size = (1, 1, Nz), extent = (1, 1, 256), topology = (Periodic, Periodic, Bounded))
grid_Les = RectilinearGrid(size = (1, 1, 256), extent = (1, 1, 256), topology = (Periodic, Periodic, Bounded))

T_init = file["timeseries/T/0"][:, :, 4:end-3]
T_Les  = CenterField(grid_Les)
set!(T_Les, T_init)

using Oceananigans.Fields: interpolate

T_mine = zeros(1, 1, Nz)
for k in 1:Nz
    T_mine[1, 1, k] = interpolate(T_Les, Center(), Center(), Center(), grid_Les, grid.xᶜᵃᵃ[1], grid.yᵃᶜᵃ[1], grid.zᵃᵃᶜ[k])
end

Tbcs = FieldBoundaryConditions(top = FluxBoundaryCondition(top_T_flux))

using Oceananigans.TurbulenceClosures: MassFluxVerticalDiffusivity

mf = MassFluxVerticalDiffusivity()
ca = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

model = HydrostaticFreeSurfaceModel(grid = grid, 
                                    buoyancy = buoyancy,
                                    velocities = PrescribedVelocityFields(),
                                    tracers = :T,
                                    boundary_conditions = (; T = Tbcs),
                                    closure = mf)

set!(model, T=T_mine)

using Oceananigans.Units
Δt = 5minutes

for i in 1:6
    for k in 1:100
        time_step!(model, Δt)
    end
    @info "we are at time step $(i * 100)"
end
    
grid_new = RectilinearGrid(size = (1, 1, 50), extent = (1, 1, 1000), topology = (Periodic, Periodic, Bounded))

model = HydrostaticFreeSurfaceModel(grid = grid_new, 
                                    buoyancy = buoyancy,
                                    velocities = PrescribedVelocityFields(),
                                    tracers = :T,
                                    boundary_conditions = (; T = Tbcs),
                                    closure = mf)

set!(model, T = (x, y, z) -> 3 + (z / 1000))                                    