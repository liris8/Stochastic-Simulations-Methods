module Stochastic_Differential_Equations

export Euler_Maruyama, Heun, Heun_ou, Heun_ou_trajs, nth_order

"""
Calculate nth order moment or centered nth order moment of trajectories.

Parameters:
- trajectories (Array): Array of trajectories.
- n (Int): Order of the moment.
- m (Int, optional): Order of centering. Default is 1.

Returns:
- If m == 1: Array containing nth order moment.
- If m > 1: Tuple containing nth order moment and m-centered nth order moment.
"""
function nth_order(trajectories, n, m = 1)
    average(trajectories) = mean(trajectories, dims=1)
    n_moment = average(trajectories .^ n) # element wise exponentiation
    if m == 1
        return n_moment[:]
    end 
    centered = trajectories .- average(trajectories)
    m_centered_moment = average(centered .^ n)
    return n_moment[:], m_centered_moment[:]
end

"""
Perform Euler-Maruyama numerical integration for stochastic differential equations.

Parameters:
- x0: Initial condition.
- t0: Initial time.
- tf: Final time.
- h: Time step for integration.
- Δt: Time step for storing data.
- D: Diffusion coefficient.
- q: Function defining the drift term.
- ntraj: Number of trajectories to simulate.

Returns:
- trajectories: Array of trajectories.
- ts: Array of time values.
"""
function Euler_Maruyama(x0, t0, tf, h, Δt, D, q, ntraj)
    n_save = round(Int, (tf - t0) / Δt) # storing times
    n_step = Int(Δt÷h) # integration steps for storage
    hD = sqrt(D * h) # avoiding sqrt calculations
    ts = [t0 + i*Δt for i in 0:(n_save)] # time values 
    trajectories = zeros(ntraj, n_save + 1) # storing 
    for tr in 1:ntraj
        x = x0; t = t0 # initialize variables
        trajectories[tr,1] = x
        for i_save in 1:n_save # loop over times at which data is stored
            for i_step in 1:n_step # inner loop integrating with timestep h
                x += h*q(x,t) + hD * randn() # new value via Euler Maruyama algorithm
                t += h # increase time by h
            end
            trajectories[tr, i_save + 1] = x # save x
        end
    end 
    return trajectories, ts
end

"""
Perform Heun's method numerical integration for ordinary differential equations with stochastic terms.

Parameters:
- q: Function defining the deterministic part of the system.
- g: Function defining the stochastic part of the system.
- h: Time step for integration.
- t0: Initial time.
- tf: Final time.
- Δt: Time step for storing data.
- x0: Initial condition.

Returns:
- ts: Array of time values.
- trajectory: Array of trajectory values.
"""
function Heun(q, g, h, t0, tf, Δt, x0)
    n_save = floor(Int, (tf - t0) / Δt) # number of times the data is saved
    n_step = floor(Int, Δt / h) # number of integration steps between each saving step

    ts = [t0 + i*Δt for i in 0:(n_save)] # time values

    dim = length(x0) # dimension of the system 
    trajectory = zeros(dim, n_save + 1) # defines the storing data matrix. Rows are for each variables and columns for each saving number  

    h_sqrt = sqrt(h)
    t = t0; x = copy(x0)
    trajectory[:, 1] = x # initial condition value stored

    for save_step in 1:n_save # loop of storing data
        for step in 1:n_step # integrating loop
            uh = h_sqrt .* randn(dim) 
            k = h .* q(x, t) .+ uh .* g(x, t)
            x += 0.5 .* (k .+ h .* q(x.+k, t+h) .+ uh .* g(x.+k, t+h))
            t += h
        end
        trajectory[:, save_step + 1] = x # saving new values
    end
    return ts, trajectory
end

"""
Perform Heun's method numerical integration for Ornstein-Uhlenbeck processes.

Parameters:
- q: Function defining the deterministic part of the system.
- g: Function defining the stochastic part of the system.
- h: Time step for integration.
- t0: Initial time.
- tf: Final time.
- Δt: Time step for storing data.
- x0: Initial condition.
- τ: Characteristic time scale of the Ornstein-Uhlenbeck process.

Returns:
- ts: Array of time values.
- trajectory: Array of trajectory values.
"""
function Heun_ou(q::Function, g::Function, h::Float64, t0::Real, tf::Real, Δt::Float64, x0::Real, τ::Float64)
    n_save = floor(Int, (tf - t0) / Δt) # number of times the data is saved
    n_step = floor(Int, Δt / h) # number of integration steps between each saving step

    ts = zeros(n_save + 1)
    trajectory = zeros(n_save + 1) # defines the storing data array. 

    # Parameters definitions
    p = exp(- h / τ)
    α = sqrt(h); β = τ * (p - 1) / α ; γ = sqrt(τ/2 * (1 - p^2) - β^2)

    # Initialize recurrence for gh
    a_old = 0; b_old = sqrt(τ/2) * (p - 1) * randn(); gh_old = 0

    # Initialize Results
    t = t0; x = x0
    ts[1] = t # initial time value stored
    trajectory[1] = x # initial condition value stored

    for save_step in 1:n_save # loop of storing data
        for _ in 1:n_step # integrating loop
            u = randn(); a = α * u; b = β * u + γ * randn()
            gh = p * (gh_old - a_old) + a - b_old + b # recurrence for gh
            a_old = a; b_old = b; gh_old = gh # updating recurrence
            k = h * q(x, t) + gh * g(x, t)
            x += 0.5 * (k + h * q(x + k, t + h) + gh * g(x + k, t + h))
            t += h
        end
        ts[save_step + 1] = t # saving new time value
        trajectory[save_step + 1] = x # saving new values 
    end
    return ts, trajectory
end

"""
Generate multiple trajectories using Heun's method for Ornstein-Uhlenbeck processes.

Parameters:
- q: Function defining the deterministic part of the system.
- g: Function defining the stochastic part of the system.
- h: Time step for integration.
- t0: Initial time.
- tf: Final time.
- Δt: Time step for storing data.
- x0: Initial condition.
- τ: Characteristic time scale of the Ornstein-Uhlenbeck process.
- ntraj: Number of trajectories to generate.

Returns:
- ts: Array of time values.
- trajectories: Array of trajectory values for each generated trajectory.
"""
function Heun_ou_trajs(q::Function, g::Function, h::Float64, t0::Real, tf::Real, Δt::Float64, x0::Real, τ::Float64, ntraj)

    n_save = floor(Int, (tf - t0) / Δt) # number of times the data is saved
    trajectories = zeros(ntraj, n_save + 1)
    ts = [t0 + i*Δt for i in 0:(n_save)]

    for traj in 1:ntraj
        _, trajectories[traj, :] = Heun_ou(q, g, h, t0, tf, Δt, x0, τ)
    end

    return ts, trajectories
end

end

