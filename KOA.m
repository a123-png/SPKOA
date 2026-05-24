function [Sun_Score, Sun_Pos, Convergence_curve] = KOA(pop, M, c, d, dim, fobj)
% KOA with full Kepler-inspired parameters and fitness update logic

% 初始化
Sun_Pos = zeros(1, dim);
Sun_Score = inf;
Convergence_curve = zeros(1, M);

% === 常数参数 ===
Tc = 3;            % 控制周期性扰动节奏
M0 = 0.1;          % 初始质量参数
lambda = 15;       % 质量衰减速率
orbital = rand(1, pop);       % 轨道偏心率
T = abs(randn(1, pop));       % 轨道周期

% 初始化位置与适应度
Positions = initialization(pop, dim, d, c);
PL_Fit = zeros(1, pop);
for i = 1:pop
    PL_Fit(i) = fobj(Positions(i, :));
    if PL_Fit(i) < Sun_Score
        Sun_Score = PL_Fit(i);
        Sun_Pos = Positions(i, :);
    end
end

% 主迭代
for t = 1:M
    worstFitness = max(PL_Fit);
    Mt = M0 * exp(-lambda * (t / M));  % 当前质量

    R = sqrt(sum((Positions - Sun_Pos).^2, 2));
    Rnorm = (R - min(R)) ./ (max(R) - min(R) + eps);

    sum_fit = sum(PL_Fit - worstFitness + eps);
    for i = 1:pop
        MS(i) = rand * (Sun_Score - worstFitness) / sum_fit;
        m(i) = (PL_Fit(i) - worstFitness) / sum_fit;
    end
    MSnorm = (MS - min(MS)) ./ (max(MS) - min(MS) + eps);
    Mnorm = (m - min(m)) ./ (max(m) - min(m) + eps);
    Fg = orbital .* Mt .* ((MSnorm .* Mnorm) ./ (Rnorm.^2 + eps)) + rand(1, pop);

    for i = 1:pop
        prev_pos = Positions(i, :);
        prev_fit = PL_Fit(i);

        % Kepler扰动参数
        a1 = rand * (T(i)^2 * (Mt * (MS(i) + m(i)) / (4 * pi^2)))^(1 / 3);
        a2 = -1 + -1 * mod(t, M / Tc) / (M / Tc);  % 周期扰动
        n = (a2 - 1) * rand + 1;

        % 随机扰动方向
        a = randi(pop); b = randi(pop);
        rd = rand(1, dim); r = rand;
        U1 = rd < r;

        if rand < rand  % 扰动方案 1：中位点扰动
            h = 1 / exp(n * randn);
            Xm = mean([Positions(b, :); Sun_Pos; Positions(i, :)], 1);
            Positions(i, :) = Positions(i, :) .* U1 + ...
                (Xm + h * (Xm - Positions(a, :))) .* (1 - U1);
        else  % 扰动方案 2：引力扰动
            f = sign(rand - 0.5);
            L = sqrt(Mt * (MS(i) + m(i)) * abs(2 / (R(i) + eps) - 1 / (a1 + eps)));
            U = rd > rand(1, dim);

            if Rnorm(i) < 0.5  % 距离较近，细粒度扰动
                Mval = rand * (1 - r) + r;
                l = L * Mval * U;
                Mv = rand * (1 - rd) + rd;
                l1 = L .* Mv .* (1 - U);
                V = l .* (2 * rand * Positions(i, :) - Positions(a, :)) + ...
                    l1 .* (Positions(b, :) - Positions(a, :)) + ...
                    (1 - Rnorm(i)) * f * U1 .* rand(1, dim) .* (d - c);
            else  % 距离远，粗粒度扰动
                U2 = rand > rand;
                V = rand * L .* (Positions(a, :) - Positions(i, :)) + ...
                    (1 - Rnorm(i)) * f * U2 * rand(1, dim) .* (rand * d - c);
            end
            Positions(i, :) = (Positions(i, :) + V * f) + ...
                (Fg(i) + abs(randn)) * U .* (Sun_Pos - Positions(i, :));
        end

        % 边界处理
        Positions(i, :) = max(min(Positions(i, :), d), c);

        % 评估 + 接受判断
        new_fit = fobj(Positions(i, :));
        if new_fit < prev_fit
            PL_Fit(i) = new_fit;
            if new_fit < Sun_Score
                Sun_Score = new_fit;
                Sun_Pos = Positions(i, :);
            end
        else
            Positions(i, :) = prev_pos;
        end
    end

    % 记录收敛曲线
    Convergence_curve(t) = Sun_Score;
end
end

% 初始化函数
function Positions = initialization(pop, dim, ub, lb)
    Positions = rand(pop, dim) .* (ub - lb) + lb;
end
