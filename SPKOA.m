function [Sun_Score, Sun_Pos, Convergence_curve] = SPKOA(pop, M, c, d, dim, fobj)

Sun_Pos = zeros(1, dim);
Sun_Score = inf;
Convergence_curve = zeros(1, M);

Tc = 3;
M0 = 0.1;
lambda = 15;
orbital = rand(1, pop);
T = abs(randn(1, pop));

% LHS初始化
raw = lhsdesign(pop, dim);
Positions = repmat(c, pop, 1) + raw .* repmat((d - c), pop, 1);
Positions = max(min(Positions, d), c);

PL_Fit = zeros(1, pop);
for i = 1:pop
    PL_Fit(i) = fobj(Positions(i,:));
    if PL_Fit(i) < Sun_Score
        Sun_Score = PL_Fit(i);
        Sun_Pos   = Positions(i,:);
    end
end

for t = 1:M
    w            = 0.3 + (1 - 0.3) * (t/M)^2;
    worstFitness = max(PL_Fit);
    Mt           = M0 * exp(-lambda * (t/M));

    R     = vecnorm(Positions - Sun_Pos, 2, 2);
    Rnorm = (R - min(R)) ./ (max(R) - min(R) + eps);

    sum_fit = sum(PL_Fit - worstFitness + eps);

    MS = rand(1, pop) .* (Sun_Score - worstFitness) / sum_fit;
    m  = (PL_Fit - worstFitness) / sum_fit;

    MSnorm = (MS - min(MS)) ./ (max(MS) - min(MS) + eps);
    Mnorm  = (m  - min(m))  ./ (max(m)  - min(m)  + eps);
    Fg     = orbital .* Mt .* ((MSnorm .* Mnorm) ./ (Rnorm.^2 + eps)) + rand(1, pop);

    alpha_t = 0.05 * (1 - t/M);
    delta   = 0.05;

    K   = min(6, pop);
    w_k = exp(-(1:K));
    w_k = w_k / sum(w_k);

    for i = 1:pop
        prev_pos = Positions(i,:);
        prev_fit = PL_Fit(i);

        a1 = rand * (T(i)^2 * (Mt * (MS(i) + m(i)) / (4*pi^2)))^(1/3);
        a2 = -1 + -1 * mod(t, M/Tc) / (M/Tc);
        n  = (a2 - 1) * rand + 1;

        a = randi(pop); b = randi(pop);
        rd = rand(1, dim); r = rand;
        U1 = rd < r;

        if rand < w
            h_scale = 1.5 - 1.2*(t/M);
            r = rand;
            if r > 0.95
                beta      = 1.0 + 1.5*(t/M);
                step_levy = levy(dim, beta);
                Xm        = (Positions(b,:) + Sun_Pos + Positions(i,:)) / 3;
                Positions(i,:) = Positions(i,:) .* U1 + ...
                    (Xm + h_scale * step_levy) .* (1 - U1);
            else
                h  = 1 / exp(n * randn);
                Xm = (Positions(b,:) + Sun_Pos + Positions(i,:)) / 3;
                Positions(i,:) = Positions(i,:) .* U1 + ...
                    (Xm + h * (Xm - Positions(a,:))) .* (1 - U1);
            end
        else
            f = sign(rand - 0.5);
            L = sqrt(Mt * (MS(i) + m(i)) * abs(2/(R(i)+eps) - 1/(a1+eps)));
            U = rd > rand(1, dim);

            if Rnorm(i) < 0.5
                Mval = rand * (1-r) + r;
                l    = L * Mval * U;
                Mv   = rand * (1-rd) + rd;
                l1   = L .* Mv .* (1-U);
                V = l  .* (2*rand*Positions(i,:) - Positions(a,:)) + ...
                    l1 .* (Positions(b,:) - Positions(a,:)) + ...
                    (1-Rnorm(i)) * f * U1 .* rand(1,dim) .* (d-c);
            else
                U2 = rand > rand;
                V  = rand * L .* (Positions(a,:) - Positions(i,:)) + ...
                     (1-Rnorm(i)) * f * U2 * rand(1,dim) .* (rand*d - c);
            end

            % 多精英引力机制
            Elite_Pos        = zeros(K, dim);
            Elite_Pos(1,:)   = Sun_Pos;
            idx              = randperm(pop, K-1);
            Elite_Pos(2:K,:) = Positions(idx,:);
            D = w_k * (Elite_Pos - Positions(i,:));

            Positions(i,:) = (Positions(i,:) + V*f) + ...
                (Fg(i) + abs(randn)) * U .* D;
        end

        Positions(i,:) = max(min(Positions(i,:), d), c);

        new_fit = fobj(Positions(i,:));

        % 先判断主更新
        if new_fit < prev_fit
            PL_Fit(i) = new_fit;
            if new_fit < Sun_Score
                Sun_Score = new_fit;
                Sun_Pos   = Positions(i,:);
            end
        else
            Positions(i,:) = prev_pos;
            PL_Fit(i)      = prev_fit;
        end

        % 再在当前位置基础上做局部微调
        sub_candidate = Positions(i,:) + alpha_t*randn(1,dim) + ...
                        delta*(Sun_Pos - Positions(i,:));
        sub_candidate = max(min(sub_candidate, d), c);
        sub_fit = fobj(sub_candidate);
        if sub_fit < PL_Fit(i)
            Positions(i,:) = sub_candidate;
            PL_Fit(i)      = sub_fit;
            if sub_fit < Sun_Score
                Sun_Score = sub_fit;
                Sun_Pos   = sub_candidate;
            end
        end
    end

    % 精英交叉
    N  = min(10, pop);
    [~, sortIdx] = sort(PL_Fit);
    elite_pool   = Positions(sortIdx(1:N), :);
    p1 = randi(N); p2 = randi(N);
    while p2 == p1; p2 = randi(N); end
    alpha     = rand;
    child     = alpha * elite_pool(p1,:) + (1-alpha) * elite_pool(p2,:);
    child     = max(min(child, d), c);
    fit_child = fobj(child);
    [~, worstIdx] = max(PL_Fit);
    if fit_child < PL_Fit(worstIdx)
        Positions(worstIdx,:) = child;
        PL_Fit(worstIdx)      = fit_child;
        if fit_child < Sun_Score
            Sun_Score = fit_child;
            Sun_Pos   = child;
        end
    end

    Convergence_curve(t) = Sun_Score;
end
end

function step = levy(dim, beta)
    sigma_u = (gamma(1+beta) * sin(pi*beta/2) / ...
              (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta);
    u    = randn(1, dim) * sigma_u;
    v    = randn(1, dim);
    step = u ./ (abs(v).^(1/beta));
end