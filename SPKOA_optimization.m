function [best_f, best_pos, curve] = SPKOA_optimization(nPop, Tmax, ub, lb, dim, fobj)


Sun_Pos   = zeros(1, dim);
Sun_Score = inf;
curve     = zeros(1, Tmax);

numMiddlePoints = fobj.numMiddlePoints;
terrain_z_func  = fobj.terrain_z;
buffer          = 0.5;


tx = fobj.terrain.x;
ty = fobj.terrain.y;
tz = fobj.terrain.z;


    function pos = apply_terrain(pos)
        pts = reshape(pos, 3, numMiddlePoints)';   
        zt  = safe_interp2(tx, ty, tz, pts(:,1), pts(:,2));  
        zt(isnan(zt)) = 0;
        pts(:,3) = max(pts(:,3), zt + buffer);
        pos = real(reshape(pts', 1, []));
        pos = min(max(pos, lb), ub);
    end


Tc      = 3;
M0      = 0.1;
lambda  = 15;
orbital = rand(1, nPop);
T       = abs(randn(1, nPop));


Positions = zeros(nPop, dim);
for i = 1:nPop
    pts = rand(numMiddlePoints, 3);
    for j = 1:3
        pts(:,j) = lb(j) + pts(:,j) * (ub(j) - lb(j));
    end
    pos_i = real(reshape(pts', 1, []));
    Positions(i,:) = apply_terrain(pos_i);
end


PL_Fit = zeros(1, nPop);
for i = 1:nPop
    PL_Fit(i) = fitness_bspline_path_soft(Positions(i,:)', fobj);
    if PL_Fit(i) < Sun_Score
        Sun_Score = PL_Fit(i);
        Sun_Pos   = Positions(i,:);
    end
end


for t = 1:Tmax

    w            = 0.3 + (1 - 0.3) * (t/Tmax)^2;
    worstFitness = max(PL_Fit);
    Mt           = M0 * exp(-lambda * (t/Tmax));

    R     = vecnorm(Positions - Sun_Pos, 2, 2);
    Rnorm = (R - min(R)) ./ (max(R) - min(R) + eps);

    sum_fit = sum(PL_Fit - worstFitness + eps);
    MS      = rand(1, nPop) .* (Sun_Score - worstFitness) / sum_fit;
    m       = (PL_Fit - worstFitness) / sum_fit;

    MSnorm = (MS - min(MS)) ./ (max(MS) - min(MS) + eps);
    Mnorm  = (m  - min(m))  ./ (max(m)  - min(m)  + eps);
    Fg     = orbital .* Mt .* ((MSnorm .* Mnorm) ./ (Rnorm.^2 + eps)) + rand(1, nPop);

    alpha_t = 0.05 * (1 - t/Tmax);
    delta   = 0.05;

    K   = min(6, nPop);
    w_k = exp(-(1:K));
    w_k = w_k / sum(w_k);

    for i = 1:nPop
        prev_pos = Positions(i,:);
        prev_fit = PL_Fit(i);

        a1 = rand * (T(i)^2 * (Mt * (MS(i) + m(i)) / (4*pi^2)))^(1/3);
        a2 = -1 + -1 * mod(t, Tmax/Tc) / (Tmax/Tc);
        n  = (a2 - 1) * rand + 1;

        a = randi(nPop); b = randi(nPop);
        rd = rand(1, dim); r = rand;
        U1 = rd < r;

        if rand < w
            h_scale = 1.5 - 1.2*(t/Tmax);
            r_tmp   = rand;
            if r_tmp > 0.95
                beta      = 1.0 + 1.5*(t/Tmax);
                step_levy = levy_flight(dim, beta);
                Xm        = (Positions(b,:) + Sun_Pos + Positions(i,:)) / 3;
                new_Pos   = Positions(i,:) .* U1 + ...
                            (Xm + h_scale * step_levy) .* (1 - U1);
            else
                h       = 1 / exp(n * randn);
                Xm      = (Positions(b,:) + Sun_Pos + Positions(i,:)) / 3;
                new_Pos = Positions(i,:) .* U1 + ...
                          (Xm + h * (Xm - Positions(a,:))) .* (1 - U1);
            end
        else
            f = sign(rand - 0.5);
            L_val = Mt * (MS(i) + m(i)) * abs(2/(R(i)+eps) - 1/(a1+eps));
            L     = sqrt(max(0, L_val));
            U     = rd > rand(1, dim);

            if Rnorm(i) < 0.5
                Mval = rand * (1-r) + r;
                l    = L * Mval * U;
                Mv   = rand * (1-rd) + rd;
                l1   = L .* Mv .* (1-U);
                V    = l  .* (2*rand*Positions(i,:) - Positions(a,:)) + ...
                       l1 .* (Positions(b,:) - Positions(a,:)) + ...
                       (1-Rnorm(i)) * f * U1 .* rand(1,dim) .* (ub - lb);
            else
                U2 = rand > rand;
                V  = rand * L .* (Positions(a,:) - Positions(i,:)) + ...
                     (1-Rnorm(i)) * f * U2 * rand(1,dim) .* (rand*ub - lb);
            end

            Elite_Pos        = zeros(K, dim);
            Elite_Pos(1,:)   = Sun_Pos;
            idx              = randperm(nPop, K-1);
            Elite_Pos(2:K,:) = Positions(idx,:);
            D = w_k * (Elite_Pos - Positions(i,:));

            new_Pos = (Positions(i,:) + V*f) + ...
                      (Fg(i) + abs(randn)) * U .* D;
        end


        new_Pos = real(new_Pos);
        new_Pos = min(max(new_Pos, lb), ub);
        new_Pos = apply_terrain(new_Pos);   

        new_fit = fitness_bspline_path_soft(new_Pos', fobj);
        if new_fit < prev_fit
            Positions(i,:) = new_Pos;
            PL_Fit(i)      = new_fit;
            if new_fit < Sun_Score
                Sun_Score = new_fit;
                Sun_Pos   = new_Pos;
            end
        else
            Positions(i,:) = prev_pos;
            PL_Fit(i)      = prev_fit;
        end

        sub_candidate = Positions(i,:) + alpha_t * randn(1,dim) + ...
                        delta * (Sun_Pos - Positions(i,:));
        sub_candidate = real(sub_candidate);
        sub_candidate = min(max(sub_candidate, lb), ub);
        sub_candidate = apply_terrain(sub_candidate); 

        sub_fit = fitness_bspline_path_soft(sub_candidate', fobj);
        if sub_fit < PL_Fit(i)
            Positions(i,:) = sub_candidate;
            PL_Fit(i)      = sub_fit;
            if sub_fit < Sun_Score
                Sun_Score = sub_fit;
                Sun_Pos   = sub_candidate;
            end
        end
    end

    N_elite = min(10, nPop);
    [~, sortIdx]     = sort(PL_Fit);
    elite_pool       = Positions(sortIdx(1:N_elite), :);
    p1 = randi(N_elite);
    p2 = randi(N_elite);
    while p2 == p1; p2 = randi(N_elite); end

    alpha_cross = rand;
    child       = alpha_cross * elite_pool(p1,:) + (1-alpha_cross) * elite_pool(p2,:);
    child       = real(child);
    child       = min(max(child, lb), ub);
    child       = apply_terrain(child);  

    fit_child = fitness_bspline_path_soft(child', fobj);
    [~, worstIdx] = max(PL_Fit);
    if fit_child < PL_Fit(worstIdx)
        Positions(worstIdx,:) = child;
        PL_Fit(worstIdx)      = fit_child;
        if fit_child < Sun_Score
            Sun_Score = fit_child;
            Sun_Pos   = child;
        end
    end

    curve(t) = Sun_Score;

end

best_f   = Sun_Score;
best_pos = Sun_Pos;
end


function step = levy_flight(dim, beta)
    sigma_u = (gamma(1+beta) * sin(pi*beta/2) / ...
              (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta);
    u    = randn(1, dim) * sigma_u;
    v    = randn(1, dim);
    v(abs(v) < eps) = eps;
    step = u ./ (abs(v).^(1/beta));
    step = real(step);
    step = min(max(step, -10), 10);
end