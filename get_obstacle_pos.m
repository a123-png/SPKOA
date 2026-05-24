function pos = get_obstacle_pos(obs, t)

    switch obs.type
        case 'linear'

            if isfield(obs, 't_end')
                t_eff = min(t, obs.t_end);
            else
                t_eff = t;
            end
            pos = obs.pos0 + obs.vel * t_eff;

        case 'circular'
            R      = norm(obs.pos0(1:2) - obs.center(1:2));
            theta0 = atan2(obs.pos0(2)-obs.center(2), obs.pos0(1)-obs.center(1));
            theta  = theta0 + obs.omega * t;
            pos    = [obs.center(1) + R*cos(theta), ...
                      obs.center(2) + R*sin(theta), ...
                      obs.center(3)];
        otherwise
            warning('get_obstacle_pos: "%s"', obs.type);
            pos = obs.pos0;
    end


    if isfield(obs, 'x_lim')
        pos(1) = min(max(pos(1), obs.x_lim(1)), obs.x_lim(2));
    end
    if isfield(obs, 'y_lim')
        pos(2) = min(max(pos(2), obs.y_lim(1)), obs.y_lim(2));
    end
    if isfield(obs, 'z_lim')
        pos(3) = min(max(pos(3), obs.z_lim(1)), obs.z_lim(2));
    end
end