"""
The main function of the project.

# Arguments
- `rng`: Random number generator. By changing the seed, we get a different lane layout.
- `sim_step`: Number of steps to simulate.
- `timestep`: timestep
- `traj_length`: the number of steps to generate for future trajectory.
- `lane_width`: width of lane
- `lane_length`: length of lane
- `num_obstacles`: number of obstacles on the lane
- `num_resources`: number of resources on the lane
- `min_r`: minimum radius of obstacles
- `max_r`: maximum radius of obstacles
- `max_vel`: maximum velocity
"""
function simulate(;
        rng = MersenneTwister(6),
        sim_steps = 63, 
        timestep = 0.2, 
        traj_length = 20,
        lane_width = 30, 
        lane_length = 120, 
        num_obstacles = 20, 
        num_resources = 5,
        min_r = 0.3, 
        max_r = 2.5, 
        max_vel = 12)

    # This array stores simulation records.
    sim_records = []

    # Defines half spaces representing the lane
    a¹ = [0; 1]; b¹ = -lane_width / 2.0
    a² = [0; -1]; b² = -lane_width / 2.0

    # Initialize car state
    car = (; state=[0.0, 0.0, 0.0, 0.0], r=1.0)

    # Generate obstacles and resources. Make sure they do not collide with each other.
    objects = [generate_object(rng, car.r, lane_width, lane_length, min_r, max_r),]
    while length(objects) < num_obstacles + num_resources
        obj = generate_object(rng, car.r, lane_width, lane_length, min_r, max_r)
        if any(collision_constraint(obj.state, obj2.state, obj.r, obj2.r) < (car.r * 2)^2 for obj2 in objects)
            continue
        else
            push!(objects, obj)
        end
    end
    obstacles = objects[1:num_obstacles]
    resources = objects[num_obstacles+1:num_obstacles+num_resources]

    @showprogress for t = 1:sim_steps

        # Get visible obstacles and objects.
        vis_obstacles = get_visible(car, obstacles, lane_width)
        vis_resources = get_visible(car, resources, lane_width)
        
        # Generate symbolic functions for the IPOPT solver.
        callbacks = create_callback_generator(length(vis_obstacles), length(vis_resources), traj_length, timestep, max_vel)
        
        # Generate trajectory
        trajectory = generate_trajectory(car, vis_obstacles, vis_resources, a¹, b¹, a², b², callbacks, traj_length)  

        # Record simulation information, including car state, trajectory, visible obstacles and visible resources
        push!(sim_records, (; car=car, trajectory, vis_obstacles, vis_resources))

        # Move the car to the next step.
        car = (; state = trajectory.states[1], r=car.r)
        
    end

    # Visualize the simulation process
    visualize_simulation(sim_records, obstacles, resources, a¹, b¹, a², b²,lane_length)
end

"""
    generate_object(rng, car_r, lane_width, lane_length, min_r, max_r)
Generate an object on the lane and make sure that it is not too close to the starting line and the
end line.
# Arguments
- `rng`: random number generator
- `car_r`: the radius of the car
- `lane_width`: width of lane
- `lane_length`: length of lane
- `min_r`: minimum radius of obstacles
- `max_r`: maximum radius of obstacles

# Returns
The state of the object.
"""
function generate_object(rng, car_r, lane_width, lane_length, min_r, max_r)
    r = min_r + rand(rng) * (max_r - min_r)
    x = 2 * car_r + r + rand(rng)*(lane_length - 4 * car_r - 2 * r)
    y = -lane_width / 2 + r + rand(rng) * (lane_width - 2 * r)
    (; state=[x,y], r)
end


"""
    generate_trajectory(car, vis_obstacles, vis_resources, a¹, b¹, a², b², callbacks, trajectory_length)

Generate a trajectory for the car.

# Arguments
- `car`: state of the car
- `vis_obstacles`: a list of visible obstacles
- `vis_resources`: a list of visible resources
- `a¹, b¹, a², b²`: parameters defining two half spaces representing the lane
- `callbacks`: the symbolic functions and constraint boundaries for the IPOPT solver
- `traj_length`: length of the trajectory

# Returns
The generated trajectory. States and Controls.
"""
function generate_trajectory(car, vis_obstacles, vis_resources, a¹, b¹, a², b², callbacks, trajectory_length)
    state = car.state
    r = car.r
    if length(vis_obstacles) != 0
        obs_states = reduce(vcat, [i.state for i in vis_obstacles])
        obs_r = [i.r for i in vis_obstacles]
    else
        obs_states = [0]
        obs_r = [0]
    end

    if length(vis_resources) != 0
        res_states = reduce(vcat, [i.state for i in vis_resources])
        res_r = [i.r for i in vis_resources]
    else
        res_states = [0]
        res_r = [0]
    end

    # Turn the symbolic functions into functions using real values.
    wrapper_f = function(z)
        callbacks.full_cost_fn(z, state, r, obs_states, obs_r, res_states, res_r, a¹, b¹, a², b²)
    end
    wrapper_grad_f = function(z, grad)
        callbacks.full_cost_grad_fn(grad, z, state, r, obs_states, obs_r, res_states, res_r, a¹, b¹, a², b²)
    end
    wrapper_con = function(z, con)
        callbacks.full_constraint_fn(con, z, state, r, obs_states, obs_r, res_states, res_r, a¹, b¹, a², b²)
    end
    wrapper_con_jac = function(z, rows, cols, vals)
        
        if isnothing(vals)
            rows .= callbacks.full_constraint_jac_triplet.jac_rows
            cols .= callbacks.full_constraint_jac_triplet.jac_cols
        else
            callbacks.full_constraint_jac_triplet.full_constraint_jac_vals_fn(vals, z, state, r, obs_states, obs_r, res_states, res_r, a¹, b¹, a², b²)
            
        end
        nothing
    end
    wrapper_lag_hess = function(z, rows, cols, cost_scaling, λ, vals)
        if isnothing(vals)
            rows .= callbacks.full_lag_hess_triplet.hess_rows
            cols .= callbacks.full_lag_hess_triplet.hess_cols
        else
            callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, z, state, r, obs_states, obs_r, res_states, res_r, a¹, b¹, a², b², λ, cost_scaling)
        end
        nothing
    end
    
    # Uses the IPOPT solver to generate trajectory.
    n = trajectory_length * 6
    m = length(callbacks.constraints_lb)
    prob = Ipopt.CreateIpoptProblem(
        n,
        fill(-Inf, n),
        fill(Inf, n),
        length(callbacks.constraints_lb),
        callbacks.constraints_lb,
        callbacks.constraints_ub,
        length(callbacks.full_constraint_jac_triplet.jac_rows),
        length(callbacks.full_lag_hess_triplet.hess_rows),
        wrapper_f,
        wrapper_con,
        wrapper_grad_f,
        wrapper_con_jac,
        wrapper_lag_hess
    )

    # Initial values.
    controls = repeat([zeros(2),], trajectory_length)
    states = repeat([state,], trajectory_length)
    zinit = compose_trajectory(states, controls)
    prob.x = zinit

    Ipopt.AddIpoptIntOption(prob, "print_level", 0)
    
    status = Ipopt.IpoptSolve(prob)

    if status != 0
        @warn "Problem not cleanly solved. IPOPT status is $(status)."
    end
    states, controls = decompose_trajectory(prob.x)
    (; states, controls, status)
end


"""
Given simulation data, visualize the simulation and generate a video.
"""
function visualize_simulation(sim_results, obstacles, resources, a1, b1, a2, b2, lane_length)
    f = Figure()
    ax = f[1,1] = Axis(f, aspect = DataAspect())
    xlims!(ax, -10, lane_length+10)
    ylims!(ax, -lane_length/4, lane_length/4)
    xcoords = [-10, lane_length+10]
    # -6, -6
    ycoords1 = (b1 .- xcoords .* a1[1]) ./ a1[2]
    # 6, 6
    ycoords2 = (b2 .- xcoords .* a2[1]) ./ a2[2]
    print(ycoords1)
    print(ycoords2)
    lines!(xcoords, ycoords1, color=:black, linewidth=3)
    lines!(xcoords, ycoords2, color=:black, linewidth=3)
    lines!([0, 0], [ycoords1[1], ycoords2[1]], color=:green, linewidth=1.5)
    lines!([lane_length, lane_length], [ycoords1[2], ycoords2[2]], color=:green, linewidth=1.5)
    car_r = sim_results[1].car.r
    car_state = sim_results[1].car.state
    view_point_x = car_state[1] - 2 * sqrt(3) / 3.0 * car_r
    
    car = Observable(Point2f(sim_results[1].car.state[1], sim_results[1].car.state[2]))

    vis_corner_1 = Point2f(car_state[1] + car_r + ycoords2[1] * 4, ycoords1[2])
    vis_corner_2 = Point2f(car_state[1] + car_r + ycoords2[1] * 4, ycoords2[2])
    vis_corner_3 = Point2f(view_point_x + (ycoords2[1] - car_state[2])/tan(pi/3), ycoords1[1])
    vis_corner_4 = Point2f(view_point_x + (ycoords2[1]+ car_state[2])/tan(pi/3), ycoords2[1])
    vis_corner_5 = Point2f(view_point_x, car_state[2])

    vis_corners = Observable([vis_corner_1, vis_corner_2, vis_corner_4, vis_corner_5, vis_corner_3])
    
    obstacle_locations = [Observable(Point2f(obs.state[1], obs.state[2])) for obs in obstacles]
    resource_locations = [Observable(Point2f(res.state[1], res.state[2])) for res in resources]
    traj = [Observable(Point2f(state[1], state[2])) for state in sim_results[1].trajectory.states]
    for t in traj
        plot!(ax, t, color=:green)
    end

    car_circle = @lift(Circle($car, car_r))
    obs_circles = [@lift(Circle($obs_loc, obs.r)) for (obs_loc, obs) in zip(obstacle_locations, obstacles)]
    res_circles = [@lift(Circle($res_loc, res.r)) for (res_loc, res) in zip(resource_locations, resources)]
    poly!(ax, car_circle, color = :black)
    poly!(ax, vis_corners, color = (:yellow, 0.5))
    cur_step = Observable(0)
    num_obs = Observable(length(sim_results[1].vis_obstacles))
    num_res = Observable(length(sim_results[1].vis_resources))
    x = Observable(sim_results[1].car.state[1])
    y = Observable(sim_results[1].car.state[2])
    v = Observable(sim_results[1].car.state[3])
    θ = Observable(sim_results[1].car.state[4])
    α = Observable(sim_results[1].trajectory.controls[1][1])
    ω = Observable(sim_results[1].trajectory.controls[1][2])
    status = Observable(sim_results[1].trajectory.status)
    step_text = map(cur_step) do step
        "Step: $step"
    end
    obs_text = map(num_obs) do obs
        "#OBS: $obs"
    end
    res_text = map(num_res) do res
        "#RES: $res"
    end
    x_text = map(x) do state_x
        "x: $state_x"
    end
    y_text = map(y) do state_y
        "y: $state_y"
    end
    v_text = map(v) do state_v
        "v: $state_v"
    end
    θ_text = map(θ) do state_θ
        "θ: $state_θ"
    end
    α_text = map(α) do state_α
        "α: $state_α"
    end
    ω_text = map(ω) do control_ω
        "ω: $control_ω"
    end
    status_text = map(status) do st
        "Status: $st"
    end
    text!(ax, step_text, position = (1, 29), align = (:left, :top), fontsize = 15, color = :black)
    text!(ax, obs_text, position = (25, 29), align = (:left, :top), fontsize = 15, color = :black)
    text!(ax, res_text, position = (25, 23), align = (:left, :top), fontsize = 15, color = :black)
    text!(ax, x_text, position = (1, -17), align = (:left, :top), fontsize = 15, color = :black)
    text!(ax, y_text, position = (1, -23), align = (:left, :top), fontsize = 15, color = :black)
    text!(ax, v_text, position = (51, -17), align = (:left, :top), fontsize = 15, color = :black)
    text!(ax, θ_text, position = (51, -23), align = (:left, :top), fontsize = 15, color = :black)
    text!(ax, α_text, position = (51, 29), align = (:left, :top), fontsize = 15, color = :black)
    text!(ax, ω_text, position = (51, 23), align = (:left, :top), fontsize = 15, color = :black)
    text!(ax, status_text, position = (1, 23), align = (:left, :top), fontsize = 15, color = :black)
    for obs in obs_circles
        poly!(ax, obs, color = :blue)
    end
    for res in res_circles
        poly!(ax, res, color = :red)
    end
    record(f, "simulation.mp4", sim_results;
        framerate = 20) do sim_step 
        for (t,state) in zip(traj, sim_step.trajectory.states)
            t[] = Point2f(state[1], state[2])
        end
        car[] = Point2f(sim_step.car.state[1], sim_step.car.state[2])
        num_obs[] = length(sim_step.vis_obstacles)
        num_res[] = length(sim_step.vis_resources)
        x[] = sim_step.car.state[1]
        y[] = sim_step.car.state[2]
        v[] = sim_step.car.state[3]
        θ[] = sim_step.car.state[4]
        α[] = sim_step.trajectory.controls[1][1]
        ω[] = sim_step.trajectory.controls[1][2]
        status[] = sim_step.trajectory.status

        view_point_x = sim_step.car.state[1] - 2 * sqrt(3) / 3.0 * car_r
        
        vis_corner_1 = Point2f(sim_step.car.state[1] + car_r + ycoords2[1] * 4, ycoords1[2])
        vis_corner_2 = Point2f(sim_step.car.state[1] + car_r + ycoords2[1] * 4, ycoords2[2])
        vis_corner_3 = Point2f(view_point_x + (ycoords2[1] + sim_step.car.state[2])/tan(pi/3), ycoords1[1])
        vis_corner_4 = Point2f(view_point_x + (ycoords2[1] - sim_step.car.state[2])/tan(pi/3), ycoords2[1])
        vis_corner_5 = Point2f(view_point_x, sim_step.car.state[2])
        vis_corners[] = [vis_corner_1, vis_corner_2, vis_corner_4, vis_corner_5, vis_corner_3]
        cur_step[] += 1
        display(f)
        sleep(0.25)
    end
end
