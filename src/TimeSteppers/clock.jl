using Adapt
using Dates: AbstractTime, Nanosecond
using Oceananigans: prettytime

import Base: show
import Oceananigans: short_show

"""
    Clock{T<:Number}

Keeps track of the current `time`, `iteration` number, and time-stepping `stage`.
`stage` is updated only for multi-stage time-stepping methods.
The `time::T` can be either a number of a `DateTime` object.
"""
mutable struct Clock{T}
         time :: T
    iteration :: Int
        stage :: Int
    
    """
        Clock{T}(time, iteration, stage=1)
    
    Returns a `Clock` with time of type `T`, initialized to the first stage.
    """
    function Clock{T}(time, iteration=0, stage=1) where T
        return new{T}(time, iteration, stage)
    end
end

"""
    Clock(; time, iteration=0, stage=1)

Returns a `Clock` initialized to the zeroth iteration and first time step stage.
"""
Clock(; time, iteration=0, stage=1) = Clock{typeof(time)}(time, iteration, stage)

short_show(clock::Clock) = string("Clock(time=", prettytime(clock.time),
                                  ", iteration=", clock.iteration, ")")

Base.show(io::IO, c::Clock{T}) where T =
    println(io, "Clock{$T}: time = ", prettytime(c.time),
                    ", iteration = ", c.iteration,
                        ", stage = ", c.stage)

tick_time!(clock, Δt) = clock.time += Δt
tick_time!(clock::Clock{<:AbstractTime}, Δt) = clock.time += Nanosecond(round(Int, 1e9 * Δt))
    
function tick!(clock, Δt; stage=false)

    tick_time!(clock, Δt)

    if stage # tick a stage update
        clock.stage += 1
    else # tick an iteration and reset stage
        clock.iteration += 1
        clock.stage = 1
    end

    return nothing
end

"Adapt `Clock` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, clock::Clock) = (time=clock.time, iteration=clock.iteration, stage=clock.stage)
