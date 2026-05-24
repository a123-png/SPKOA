function [fit, L, T_flight] = fitness_bspline_path_soft(individual, fobj)

    pts = reshape(individual, 3, fobj.numMiddlePoints)';
    pts_full = [fobj.startPoint; pts; fobj.endPoint];


    num_ctrl_pts = size(pts_full, 1);
    t_ctrl   = linspace(0, 1, num_ctrl_pts);
    t_interp = linspace(0, 1, 200);
    spline_x = spline(t_ctrl, pts_full(:,1), t_interp)';
    spline_y = spline(t_ctrl, pts_full(:,2), t_interp)';
    spline_z = spline(t_ctrl, pts_full(:,3), t_interp)';
    smoothPath = [spline_x, spline_y, spline_z];
    z_vals = smoothPath(:,3);
    n_pts  = size(smoothPath, 1);


    w_length   = 60;   
    w_attitude = 25;   
    w_zrange   = 15;   



    L_ref = 300;   


    L = computePathLength(smoothPath);
    cost_length = L / L_ref;


    Phi_max = deg2rad(45); Theta_max = deg2rad(45);

    dp_ctrl = diff(pts_full, 1, 1);
    dx_ctrl = dp_ctrl(:,1); dy_ctrl = dp_ctrl(:,2); dz_ctrl = dp_ctrl(:,3);

    vec1 = [dx_ctrl(1:end-1), dy_ctrl(1:end-1)];
    vec2 = [dx_ctrl(2:end),   dy_ctrl(2:end)  ];
    n1 = sqrt(sum(vec1.^2,2)); n2 = sqrt(sum(vec2.^2,2));
    valid_v = (n1>1e-6) & (n2>1e-6);
    cos_theta = sum(vec1(valid_v,:).*vec2(valid_v,:),2)./(n1(valid_v).*n2(valid_v));
    cos_theta = min(1,max(-1,cos_theta));
    yaw = abs(acos(cos_theta));
    over_yaw = yaw > Theta_max;
    penalty_yaw = sum((yaw(over_yaw) - Theta_max).^2);

    horiz_ctrl = sqrt(dx_ctrl.^2 + dy_ctrl.^2);
    valid_h_ctrl = horiz_ctrl > 1e-6;
    pitch = abs(atan2(dz_ctrl(valid_h_ctrl), horiz_ctrl(valid_h_ctrl)));
    over_pitch = pitch > Phi_max;
    penalty_pitch = sum((pitch(over_pitch) - Phi_max).^2);

    cost_attitude = (penalty_pitch + penalty_yaw) / L_ref;

    allowed_range = 15;
    z_range = max(z_vals) - min(z_vals);
    cost_zrange = max(0, z_range - allowed_range) / allowed_range;

    cost_obj = w_length   * cost_length   + ...
               w_attitude * cost_attitude + ...
               w_zrange   * cost_zrange;


    lb = fobj.lb; ub = fobj.ub;
    x_viol = max(0, lb(1)-smoothPath(:,1)) + max(0, smoothPath(:,1)-ub(1));
    y_viol = max(0, lb(2)-smoothPath(:,2)) + max(0, smoothPath(:,2)-ub(2));
    z_viol = max(0, lb(3)-smoothPath(:,3)) + max(0, smoothPath(:,3)-ub(3));
    penalty_bound = 1e6 * sum(x_viol.^2 + y_viol.^2 + z_viol.^2);

    safety_margin = 5;
    penalty_terrain = 0;
    terrain_z = nan(n_pts, 1);
    if isfield(fobj, 'terrain')
        valid_mask = all(isfinite(smoothPath(:,1:2)), 2);
        terrain_z(valid_mask) = interp2( ...
            fobj.terrain.x, fobj.terrain.y, fobj.terrain.z, ...
            smoothPath(valid_mask,1), smoothPath(valid_mask,2), 'linear');
        invalid_t  = isnan(terrain_z);
        below_safe = smoothPath(:,3) < (terrain_z + safety_margin);
        if any(invalid_t) || any(below_safe)
            z_viol_t = (terrain_z(below_safe)+safety_margin) - smoothPath(below_safe,3);
            penalty_terrain = 1e4 * sum(z_viol_t.^2) + 1e4 * sum(invalid_t);
        end
    end

    nfz_safety = 10;
    penalty_nfz = 0;
    if isfield(fobj, 'no_fly_zones')
        px = smoothPath(:,1); py = smoothPath(:,2); pz = smoothPath(:,3);
        for j = 1:length(fobj.no_fly_zones)
            nfz     = fobj.no_fly_zones(j);
            r_hard  = nfz.radius;
            r_soft  = nfz.radius + nfz_safety;
            dist_xy = sqrt((px-nfz.cx).^2 + (py-nfz.cy).^2);
            in_h    = (pz >= nfz.z_min) & (pz <= nfz.z_max);
            hard = (dist_xy < r_hard) & in_h;
            if any(hard)
                penalty_nfz = penalty_nfz + 1e5 * sum((r_hard - dist_xy(hard)).^2);
            end
            soft = (dist_xy >= r_hard) & (dist_xy < r_soft) & in_h;
            if any(soft)
                penalty_nfz = penalty_nfz + 1e2 * sum((r_soft - dist_xy(soft)).^2);
            end
        end
    end

    dyn_safety = 15;
    penalty_dyn = 0;
    if isfield(fobj, 'obstacles') && isfield(fobj, 't_per_segment')
        uav_speed = 10;
        seg_l = sqrt(sum(diff(smoothPath).^2, 2));
        cum_l = [0; cumsum(seg_l)];
        t_vec = cum_l / uav_speed;
        for j = 1:length(fobj.obstacles)
            obs_traj = compute_obs_trajectory(fobj.obstacles(j), t_vec);
            dist_all = sqrt(sum((smoothPath - obs_traj).^2, 2));
            r       = fobj.obstacles(j).radius;
            r_soft  = r + dyn_safety;
            inside = dist_all < r;
            if any(inside)
                penalty_dyn = penalty_dyn + 1e5 * sum((r - dist_all(inside)).^2);
            end
            buffer = (dist_all >= r) & (dist_all < r_soft);
            if any(buffer)
                penalty_dyn = penalty_dyn + 1e2 * sum((r_soft - dist_all(buffer)).^2);
            end
        end
    end

    hard_penalty = penalty_bound + penalty_terrain + penalty_nfz + penalty_dyn;
    fit = cost_obj + hard_penalty;

    if hard_penalty > 0
        fit = fit * 1000;
    end

    T_flight = L / 10;  
end


function traj = compute_obs_trajectory(obs, t_vec)
    n = length(t_vec);
    switch obs.type
        case 'linear'

            if isfield(obs, 't_end')
                t_eff = min(t_vec, obs.t_end);
            else
                t_eff = t_vec;
            end
            traj = repmat(obs.pos0, n, 1) + t_eff * obs.vel;
            if isfield(obs, 'x_lim')
                traj(:,1) = min(max(traj(:,1), obs.x_lim(1)), obs.x_lim(2));
            end
            if isfield(obs, 'y_lim')
                traj(:,2) = min(max(traj(:,2), obs.y_lim(1)), obs.y_lim(2));
            end
            if isfield(obs, 'z_lim')
                traj(:,3) = min(max(traj(:,3), obs.z_lim(1)), obs.z_lim(2));
            end
        case 'circular'
            R      = norm(obs.pos0(1:2) - obs.center(1:2));
            theta0 = atan2(obs.pos0(2)-obs.center(2), obs.pos0(1)-obs.center(1));
            theta  = theta0 + obs.omega * t_vec;
            traj   = [obs.center(1) + R*cos(theta), ...
                      obs.center(2) + R*sin(theta), ...
                      repmat(obs.center(3), n, 1)];
        otherwise
            traj = repmat(obs.pos0, n, 1);
    end
end