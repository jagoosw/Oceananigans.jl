#####
##### Apparently this is required to ensure synchronization...
#####

####
#### Fill boundary buffers from Field
####

function fill_send_boundary_buffers!(buffer, c, arch, grid)

    Hx, Hy, Hz = halo_size(grid)
    Nx, Ny, Nz = size(grid)
    west_event   = fill_send_buffer_west!(buffer.west,   parent(c), Hx, Nx, arch, grid)
    east_event   = fill_send_buffer_east!(buffer.east,   parent(c), Hx, Nx, arch, grid)
    south_event  = fill_send_buffer_south!(buffer.south, parent(c), Hy, Ny, arch, grid)
    north_event  = fill_send_buffer_north!(buffer.north, parent(c), Hy, Ny, arch, grid)

    wait(device(arch), MultiEvent((west_event, east_event, south_event, north_event)))

    return nothing
end

@inline fill_send_buffer_west!(::Nothing, args...)  = NoneEvent()
@inline fill_send_buffer_east!(::Nothing, args...)  = NoneEvent()
@inline fill_send_buffer_south!(::Nothing, args...) = NoneEvent()
@inline fill_send_buffer_north!(::Nothing, args...) = NoneEvent()

@inline fill_send_buffer_west!(buffer,  c, H, N, arch, grid)  = launch!(arch, grid, size(c)[[2, 3]], _fill_send_buffer_west!,  buffer.send, c, H, N)
@inline fill_send_buffer_east!(buffer,  c, H, N, arch, grid)  = launch!(arch, grid, size(c)[[2, 3]], _fill_send_buffer_east!,  buffer.send, c, H, N)
@inline fill_send_buffer_south!(buffer, c, H, N, arch, grid)  = launch!(arch, grid, size(c)[[1, 3]], _fill_send_buffer_south!, buffer.send, c, H, N)
@inline fill_send_buffer_north!(buffer, c, H, N, arch, grid)  = launch!(arch, grid, size(c)[[1, 3]], _fill_send_buffer_north!, buffer.send, c, H, N)

@kernel function _fill_send_buffer_west!(buffer, c, H, N) 
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        @inbounds buffer[i, j, k] = c[i+H, j, k]
    end
end

@kernel function _fill_send_buffer_east!(buffer, c, H, N) 
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        @inbounds buffer[i, j, k] = c[i+N, j, k]
    end
end

@kernel function _fill_send_buffer_south!(buffer, c, H, N) 
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        @inbounds buffer[i, j, k] = c[i, j+H, k]
    end
end

@kernel function _fill_send_buffer_north!(buffer, c, H, N) 
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        @inbounds buffer[i, j, k] = c[i, j+N, k]
    end
end

#####
##### Fill halo from buffers
#####

@kernel function _fill_west_and_east_halo_from_buffer!(c, westbuff, eastbuff, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        @inbounds begin
            c[i, j, k]     = westbuff[i, j, k]
            c[i+N+H, j, k] = eastbuff[i, j, k]
        end
    end
end

@kernel function _fill_south_and_north_halo_from_buffer!(c, southbuff, northbuff, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        @inbounds begin
            c[i, j, k]     = southbuff[i, j, k]
            c[i, j+N+H, k] = northbuff[i, j, k]
        end
    end
end

@kernel function _fill_west_halo_from_buffer!(c, buff, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        @inbounds c[i, j, k] = buff[i, j, k]
    end
end

@kernel function _fill_east_halo_from_buffer!(c, buff, H, N)
    j, k = @index(Global, NTuple)
    @unroll for i in 1:H
        @inbounds c[i+N+H, j, k] = buff[i, j, k]
    end
end

@kernel function _fill_south_halo_from_buffer!(c, buff, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        @inbounds c[i, j, k] = buff[i, j, k]
    end
end

@kernel function _fill_north_halo_from_buffer!(c, buff, H, N)
    i, k = @index(Global, NTuple)
    @unroll for j in 1:H
        @inbounds c[i, j+N+H, k] = buff[i, j, k]
    end
end
