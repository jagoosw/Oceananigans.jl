include("dependencies_for_runtests.jl")

using MPI

# # Distributed model tests
#
# These tests are meant to be run on 4 ranks. This script may be run
# stand-alone (outside the test environment) via
#
# mpiexec -n 4 julia --project test_distributed_models.jl
#
# provided that a few packages (like TimesDates.jl) are in your global environment.
#
# Another possibility is to use tmpi ():
#
# tmpi 4 julia --project
#
# then later:
# 
# julia> include("test_distributed_models.jl")
#
# When running the tests this way, uncomment the following line

MPI.Init()

# to initialize MPI.

using Oceananigans.Distributed: reconstruct_global_grid
using Oceananigans.Solvers: copy_real_component!

function random_divergent_source_term(grid)
    # Generate right hand side from a random (divergent) velocity field.
    Ru = XFaceField(grid)
    Rv = YFaceField(grid)
    Rw = ZFaceField(grid)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, (x, y, z) -> rand())
    set!(Rv, (x, y, z) -> rand())
    set!(Rw, (x, y, z) -> rand())

    arch = architecture(grid)
    fill_halo_regions!(Ru)
    fill_halo_regions!(Rv)
    fill_halo_regions!(Rw)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R
end

function divergence_free_poisson_solution_triply_periodic(grid_points, ranks)
    topo = (Periodic, Periodic, Periodic)
    arch = MultiArch(CPU(), ranks=ranks, topology = topo)
    local_grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(1, 2, 3))
    
    global_grid = reconstruct_global_grid(local_grid)
    solver = DistributedFFTBasedPoissonSolver(global_grid, local_grid)

    R = random_divergent_source_term(local_grid)
    first(solver.storage) .= R

    ϕc = first(solver.storage)
    solve!(ϕc, solver)

    p_bcs = FieldBoundaryConditions(local_grid, (Center, Center, Center))
    p_bcs = inject_halo_communication_boundary_conditions(p_bcs, arch.local_rank, arch.connectivity)

    ϕ   = CenterField(local_grid, boundary_conditions=p_bcs) # "pressure"
    ∇²ϕ = CenterField(local_grid, boundary_conditions=p_bcs)

    copy_event = launch!(arch, local_grid, :xyz,
                         copy_real_component!, ϕ, first(solver.storage),
                         dependencies=device_event(arch))
    wait(device(arch), copy_event)

    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return R ≈ interior(∇²ϕ)
end

@testset "Distributed FFT-based Poisson solver" begin
    @info "  Testing distributed FFT-based Poisson solver..."
    @test divergence_free_poisson_solution_triply_periodic((16, 16, 1), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((44, 44, 1), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((44, 16, 1), (1, 4, 1))
    @test divergence_free_poisson_solution_triply_periodic((16, 44, 1), (1, 4, 1))
end
