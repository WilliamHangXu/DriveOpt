"""
Generate symbolic functions that are accepted by the IPOPT solver, and constraint bounds.

# Arguments
- `num_obs`: number of visible obstacles
- `num_res`: number of visible resources
- `trajectory_length`: the number of steps to generate for future trajectory
- `timestep`: timestep
- `max_vel`: maximum velocity
"""
function create_callback_generator(num_obs, num_res, trajectory_length, timestep, max_vel)
    # Define symbolic variables for all inputs, as well as trajectory
    X, r, a¹, b¹, a², b², Z = let
        @variables(X[1:4], r, a¹[1:2], b¹, a²[1:2], b², Z[1:6*trajectory_length]) .|> Symbolics.scalarize
    end

    # Define symbolic variables to represent obstacle states. If there's no visible obstacles, just
    # generate dummy variables.
    if num_obs != 0
        obs_state, obs_r = let 
            @variables(obs_state[1:2*num_obs], obs_r[1:num_obs]) .|> Symbolics.scalarize
        end
    else
        obs_state, obs_r = let 
            @variables(obs_state[1:1], obs_r[1:1]) .|> Symbolics.scalarize
        end
    end

    # Define symbolic variables to represent resource states. If there's no visible resources, just
    # generate dummy variables.
    if num_res != 0
        res_state, res_r = let 
            @variables(res_state[1:2*num_res], res_r[1:num_res]) .|> Symbolics.scalarize
        end
    else
        res_state, res_r = let 
            @variables(res_state[1:1], res_r[1:1]) .|> Symbolics.scalarize
        end
    end
    
    # Get states and controls from trajectory
    states, controls = decompose_trajectory(Z)
    all_states = [[X,]; states]

    # Get an array of coordinates for obstacles and resources
    if num_obs != 0
        obs_sliced = [obs_state[i:i+1] for i in 1:2:length(obs_state)]
    end
    if num_res != 0
        res_sliced = [res_state[i:i+1] for i in 1:2:length(res_state)]
    else
        res_sliced = []
    end

    # Calculate the objective function. This is the sum of stage costs.
    cost_val = sum(stage_cost(x, u, res_sliced) for (x,u) in zip(states, controls))

    # Calculate the gradient of the cost function with respect to the trajectory variables.
    cost_grad = Symbolics.gradient(cost_val, Z)

    # Get arrays for constraint variables, their upper bound and lower bound.
    constraints_val = Symbolics.Num[]
    constraints_lb = Float64[]
    constraints_ub = Float64[]
    for k in 1:trajectory_length
        # Obey physics.
        append!(constraints_val, all_states[k+1] .- evolve_state(all_states[k], controls[k], timestep))
        append!(constraints_lb, zeros(4))
        append!(constraints_ub, zeros(4))

        # Within lane boundaries
        append!(constraints_val, lane_constraint(states[k], a¹, b¹, r))
        append!(constraints_val, lane_constraint(states[k], a², b², r))
        append!(constraints_lb, zeros(2))
        append!(constraints_ub, fill(Inf, 2))

        # Dodge obstacles
        for j in 1:num_obs
            append!(constraints_val, collision_constraint(states[k][1:2], obs_sliced[j], r, obs_r[j]))
            append!(constraints_lb, 0.0)
            append!(constraints_ub, Inf)
        end

        # Speed limit
        append!(constraints_val, states[k][3])
        append!(constraints_lb, 1.0)
        append!(constraints_ub, max_vel)

        # Heading angle
        append!(constraints_val, states[k][4])
        append!(constraints_lb, -pi/3)
        append!(constraints_ub, pi/3)
    end

    constraints_jac = Symbolics.sparsejacobian(constraints_val, Z)
    (jac_rows, jac_cols, jac_vals) = findnz(constraints_jac)
    num_constraints = length(constraints_val)

    λ, cost_scaling = let
        @variables(λ[1:num_constraints], cost_scaling) .|> Symbolics.scalarize
    end

    lag = (cost_scaling * cost_val + λ' * constraints_val)
    lag_grad = Symbolics.gradient(lag, Z)
    lag_hess = Symbolics.sparsejacobian(lag_grad, Z)
    (hess_rows, hess_cols, hess_vals) = findnz(lag_hess)
    
    expression = Val{false}

    # The function that computes the objective function
    full_cost_fn = let
        cost_fn = Symbolics.build_function(cost_val, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²]; expression)
        (Z, X, r, obs_state, obs_r, res_state, res_r, a¹, b¹, a², b²) -> cost_fn([Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²])
    end

    # The function that computes the gradient of the objective function
    full_cost_grad_fn = let
        cost_grad_fn! = Symbolics.build_function(cost_grad, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²]; expression)[2]
        (grad, Z, X, r, obs_state, obs_r, res_state, res_r, a¹, b¹, a², b²) -> cost_grad_fn!(grad, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²])
    end

    # The function that computes the constraint terms
    full_constraint_fn = let
        constraint_fn! = Symbolics.build_function(constraints_val, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²]; expression)[2]
        (cons, Z, X, r, obs_state, obs_r, res_state, res_r, a¹, b¹, a², b²) -> constraint_fn!(cons, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²])
    end

    # The function that computes the jacobian of the constraint terms
    full_constraint_jac_vals_fn = let
        constraint_jac_vals_fn! = Symbolics.build_function(jac_vals, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²]; expression)[2]
        (vals, Z, X, r, obs_state, obs_r, res_state, res_r, a¹, b¹, a², b²) -> constraint_jac_vals_fn!(vals, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²])
    end
    
    # The function that computes the hessian of the lagrangian
    full_hess_vals_fn = let
        hess_vals_fn! = Symbolics.build_function(hess_vals, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²; λ; cost_scaling]; expression)[2]
        (vals, Z, X, r, obs_state, obs_r, res_state, res_r, a¹, b¹, a², b², λ, cost_scaling) -> hess_vals_fn!(vals, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²; λ; cost_scaling])
    end

    full_constraint_jac_triplet = (; jac_rows, jac_cols, full_constraint_jac_vals_fn)
    full_lag_hess_triplet = (; hess_rows, hess_cols, full_hess_vals_fn)

    return (; full_cost_fn, 
            full_cost_grad_fn, 
            full_constraint_fn, 
            full_constraint_jac_triplet, 
            full_lag_hess_triplet,
            constraints_lb,
            constraints_ub)
end


"""
    evolve_state(X, U, Δ)
Given current state and control, compute the next state.
# Arguments
- `X`: current state
- `U`: current control
- `Δ`: timestep
# Returns
- The next state.
"""
function evolve_state(X, U, Δ)
    V = X[3] + Δ * U[1] 
    θ = X[4] + Δ * U[2]
    X + Δ * [V*cos(θ), V*sin(θ), U[1], U[2]]
end

"""
    lane_constraint(X, a, b, r)
Given the status of the car and the half space representing one lane boundary, decide if the car
completely lies in the boundary. The returned value should be larger than 0 if the car is within
the boundary.

# Arguments
- `X`: the state of the car
- `a, b`: the half space
- `r`: the radius of the car
"""
function lane_constraint(X, a, b, r)
    a'*(X[1:2] - a*r)-b
end


"""
    collision_constraint(X1, X2, r1, r2)
Given two circles, compute (distance between two centers)^2 - (sum of radii)^2. This should be larger
than 0 if two circles do not touch each other.

# Arguments 
- `X1`：the coordinate of the center of the first circle
- `X2`: the coordinate of the center of the second circle
- `r1`: the radius of the first circle
- `r2`: the radius of the second circle
"""
function collision_constraint(X1, X2, r1, r2)
    (X1-X2)'*(X1-X2) - (r1+r2)^2
end


"""
    stage_cost(X, U, res)
Calculate the cost of each step. The sum of this function at each step is the objective function of
the optimization problem. For our problem set up, we want to: 
1. Make the car go fast, so we have the term -a * X[3], where a is a positive coefficient and X[3] is speed;
2. Make the car go through the resources, so we have the term -b * res_award. res_award is the sum of 
   exp(-500*distance^2) of the car and the resources. 
3. Make the car's trajectory smooth, so we have the term U' * R * U, because we want to penalize 
   change in velocity and heading angle.

# Arguments
- `X`: current state
- `U`: current control
- `R`: penalty coefficients for acceleration and angular velocity. This is a 2*2 diagonal matrix, 
        the first term is for acceleration and the second term is for angular velocity
- `res`: the array containing information about resources
# Returns
- `cost`: the cost at the current step
"""
function stage_cost(X, U, res)
    res_award = 0
    for i in res
        res_award += exp(-500 * (X[1:2]-i)'*(X[1:2]-i))
    end
    R = Diagonal([0.0, 0.01])
    cost =  -0.1 * X[3] - res_award + U' * R * U
end


"""
    decompose_trajectory(z)
Given a trajectory, get states and controls.

# Arguments
- `z`: A trajectory in the form [U[1];...;U[K];X[1];...;X[K]]
# Returns
- `states`: a list of states: [X[1], X[2],..., X[K]]
- `controls`: a list of controls: [U[1],...,U[K]]
"""
function decompose_trajectory(z)
    K = Int(length(z) / 6)
    controls = [@view(z[(k-1)*2+1:k*2]) for k = 1:K]
    states = [@view(z[2K+(k-1)*4+1:2K+k*4]) for k = 1:K]
    return states, controls
end


"""
    compose_trajectory(states, controls)
Given states and controls, get a trajectory.

# Arguments
- `states`: a list of states: [X[1], X[2],..., X[K]]
- `controls`: a list of controls: [U[1],...,U[K]]
# Returns
A trajectory in the form [U[1];...;U[K];X[1];...;X[K]]
"""
function compose_trajectory(states, controls)
    K = length(states)
    z = [reduce(vcat, controls); reduce(vcat, states)]
end


"""
    get_visible(car, objects, lane_width)
Get a list of visible objects (obstacles or resources) for the car.
The visible area looks like this:
                           ____________________
                          //       lane        |
upper vision boundary -> //                    |
                        // π/3                 |
                      car ----<2*lane_width>---|  <- horizon
                        \\ π/3                 |
lower vision boundary -> \\                    |
                          \\_______lane________|

# Arguments
- `car`: the named tuple representing the car: [state=[x, y, r, θ], r].
- `objects`: a list of objects. An object is represented by a named tuple: [state=[x, y], r].
- `lane_width`: The width of the lane. The furthest point the car can see (the horizon) is 
2 * lane_width away from its front.
# Returns
- `visible`: a list of visible objects.

"""
function get_visible(car, objects, lane_width)
    # Get the state and radius of the car
    state = car.state
    r = car.r

    # Calculate the x-coordinate of the view point (where the upper vision boundary meets the 
    # lower vision boundary)
    view_point_x = state[1] - 2 * sqrt(3) / 3.0 * r

    # Defines the half space boundary representing the upper vision boundary
    up_a = [-sqrt(3); 1]
    up_b = -sqrt(3) * view_point_x + state[2]
    
    # Defines the half space boundary representing the lower vision boundary
    down_a = [sqrt(3); 1]
    down_b = sqrt(3) * view_point_x + state[2]
    
    # The x-coordinate of the horizon
    front = state[1] + r + lane_width * 2

    # The list storing the visible objects
    visible = []

    # As long as the object lies in the visible area, put that object into visible.
    for i in objects
        up_point = [i.state[1] + sqrt(3) / 2.0 * i.r, i.state[2] - i.r / 2.0]
        down_point = [i.state[1] + sqrt(3) / 2.0 * i.r, i.state[2] + i.r / 2.0]
        if i.state[1] < front + i.r && up_a' * up_point < up_b && down_a' * down_point > down_b
            push!(visible, i)
        end
    end
    visible
end

