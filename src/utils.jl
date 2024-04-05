"""
Create functions which accepts X¹, X², X³, r¹, r², r³, a¹, b¹, a², b², as input, and each return
one of the 5 callbacks which constitute an IPOPT problem: 
1. eval_f
2. eval_g
3. eval_grad_f
4. eval_jac_g
5. eval_hess_lag

Xⁱ is the vehicle_state of vehicle i at the start of the trajectory (t=0)
rⁱ is the radius of the i-th vehicle.
(aⁱ, bⁱ) define a halfpsace representing one of the two lane boundaries. 

The purpose of this function is to construct functions which can quickly turn 
updated world information into planning problems that IPOPT can solve.
"""
function create_callback_generator(num_obs, num_res, trajectory_length=40, timestep=0.2, R = Diagonal([0.1, 0.5]), max_vel=10.0)
    # Define symbolic variables for all inputs, as well as trajectory
    X, r, a¹, b¹, a², b², Z = let
        @variables(X[1:4], r, a¹[1:2], b¹, a²[1:2], b², Z[1:6*trajectory_length]) .|> Symbolics.scalarize
    end

    if num_obs != 0
        obs_state, obs_r = let 
            @variables(obs_state[1:2*num_obs], obs_r[1:num_obs]) .|> Symbolics.scalarize
        end
    else
        obs_state, obs_r = let 
            @variables(obs_state[1:1], obs_r[1:1]) .|> Symbolics.scalarize
        end
    end

    if num_res != 0
        res_state, res_r = let 
            @variables(res_state[1:2*num_res], res_r[1:num_res]) .|> Symbolics.scalarize
        end
    else
        res_state, res_r = let 
            @variables(res_state[1:1], res_r[1:1]) .|> Symbolics.scalarize
        end
    end
    
    states, controls = decompose_trajectory(Z)
    all_states = [[X,]; states]
    if num_obs != 0
        obs_sliced = [obs_state[i:i+1] for i in 1:2:length(obs_state)]
    end
    if num_res != 0
        res_sliced = [res_state[i:i+1] for i in 1:2:length(res_state)]
    else
        res_sliced = []
    end
    cost_val = sum(stage_cost(x, u, R, res_sliced) for (x,u) in zip(states, controls))
    cost_grad = Symbolics.gradient(cost_val, Z)

    constraints_val = Symbolics.Num[]
    constraints_lb = Float64[]
    constraints_ub = Float64[]
    for k in 1:trajectory_length
        append!(constraints_val, all_states[k+1] .- evolve_state(all_states[k], controls[k], timestep))
        append!(constraints_lb, zeros(4))
        append!(constraints_ub, zeros(4))
        append!(constraints_val, lane_constraint(states[k], a¹, b¹, r))
        append!(constraints_val, lane_constraint(states[k], a², b², r))
        append!(constraints_lb, zeros(2))
        append!(constraints_ub, fill(Inf, 2))
        ########
        for j in 1:num_obs
            append!(constraints_val, collision_constraint(states[k][1:2], obs_sliced[j], r, obs_r[j]))
            append!(constraints_lb, 0.0)
            append!(constraints_ub, Inf)
        end
        ########
        append!(constraints_val, states[k][3])
        append!(constraints_lb, 1.0)
        append!(constraints_ub, max_vel)
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

    full_cost_fn = let
        cost_fn = Symbolics.build_function(cost_val, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²]; expression)
        (Z, X, r, obs_state, obs_r, res_state, res_r, a¹, b¹, a², b²) -> cost_fn([Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²])
    end

    full_cost_grad_fn = let
        cost_grad_fn! = Symbolics.build_function(cost_grad, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²]; expression)[2]
        (grad, Z, X, r, obs_state, obs_r, res_state, res_r, a¹, b¹, a², b²) -> cost_grad_fn!(grad, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²])
    end

    full_constraint_fn = let
        constraint_fn! = Symbolics.build_function(constraints_val, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²]; expression)[2]
        (cons, Z, X, r, obs_state, obs_r, res_state, res_r, a¹, b¹, a², b²) -> constraint_fn!(cons, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²])
    end

    full_constraint_jac_vals_fn = let
        constraint_jac_vals_fn! = Symbolics.build_function(jac_vals, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²]; expression)[2]
        (vals, Z, X, r, obs_state, obs_r, res_state, res_r, a¹, b¹, a², b²) -> constraint_jac_vals_fn!(vals, [Z; X; r; obs_state; obs_r; res_state; res_r; a¹; b¹; a²; b²])
    end
    
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
The physics model used for motion planning purposes.
Returns X[k] when inputs are X[k-1] and U[k]. 
Uses a slightly different vehicle model than presented in class for technical reasons.
"""
function evolve_state(X, U, Δ)
    V = X[3] + Δ * U[1] 
    θ = X[4] + Δ * U[2]
    X + Δ * [V*cos(θ), V*sin(θ), U[1], U[2]]
end

function lane_constraint(X, a, b, r)
    a'*(X[1:2] - a*r)-b
end

function collision_constraint(X1, X2, r1, r2)
    (X1-X2)'*(X1-X2) - (r1+r2)^2
end


"""
    stage_cost(X, U, R, res)
Calculates the 
"""
function stage_cost(X, U, R, res)
    res_award = 0
    for i in res
        res_award += exp(-500 * (X[1:2]-i)'*(X[1:2]-i))
    end
    cost =  -0.1 * X[3] - res_award + U' * R * U
end


"""
    decompose_trajectory(z)
Given a trajectory, get states and controls.

# Returns
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

