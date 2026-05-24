clear;
clc;
close all;
addpath(genpath(pwd));

pop_size = 50;
max_iter = 500;
run      = 30;
plot_convergence = 1;
CNT      = 20;

RESULT      = [];
TIME_RESULT = [];
PVAL_RESULT   = cell(1, 29);
WTLROW_RESULT = cell(1, 29);
RANK_DATA     = [];
WTL_RESULT    = [];

F            = [1 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30];
variables_no = 30;

fprintf('CEC2017 | Dim = %d\n\n', variables_no);

algorithms = {
    @KOA,  'KOA';
    @SPKOA, 'SPKOA';
};
num_algorithms = size(algorithms, 1);

colors = {
    [0.00  0.447 0.741];
    [0.850 0.325 0.098];
};
markers    = {'o', 's'};
line_types = {'-', '--'};

for func_num = 1:length(F)
    current_func = F(func_num);
    fprintf('Processing F%d...\n', current_func);

    [lower_bound, upper_bound, dim, fobj] = Get_Functions_cec2017(current_func, variables_no);

    resu          = [];
    rank_sum_resu = [];
    wtl_resu      = [];
    time_row      = zeros(1, num_algorithms);
    all_curves    = cell(num_algorithms, run+1);
    all_results   = zeros(num_algorithms, run);

    for alg_idx = 1:num_algorithms
        alg_handle = algorithms{alg_idx, 1};
        alg_name   = algorithms{alg_idx, 2};

        final_results = zeros(1, run);
        time_results  = zeros(1, run);
        curves        = cell(1, run);

        for nrun = 1:run
            rng(nrun * 1000 + current_func);
            tic;
            [final_score, ~, curve] = alg_handle(pop_size, max_iter, lower_bound, upper_bound, dim, fobj);
            time_results(nrun)  = toc;
            final_results(nrun) = final_score;
            curves{nrun}        = curve;
            all_curves{alg_idx, nrun} = curve;
        end

        all_results(alg_idx, :) = final_results;
        time_row(alg_idx) = mean(time_results);

        [~, best_run_idx] = min(final_results);
        all_curves{alg_idx, run+1} = curves{best_run_idx};

        stats = [min(final_results); mean(final_results); std(final_results); max(final_results)];
        resu  = [resu, stats];

        fprintf('  %-8s Min=%.4e  Mean=%.4e  Std=%.4e  Time=%.4fs\n', ...
            alg_name, stats(1), stats(2), stats(3), time_row(alg_idx));
    end

    TIME_RESULT = [TIME_RESULT; time_row];

    %% Wilcoxon rank-sum test (SPKOA as baseline)
    spkoa_idx     = 2;
    spkoa_results = all_results(spkoa_idx, :);
    spkoa_mean    = mean(spkoa_results);

    for alg_idx = 1:num_algorithms
        if alg_idx == spkoa_idx
            rank_sum_resu = [rank_sum_resu, NaN];
            wtl_resu      = [wtl_resu, {'--'}];
            continue;
        end
        other_results = all_results(alg_idx, :);
        other_mean    = mean(other_results);
        rs = perform_ranksum(spkoa_results, other_results);

        if isnan(rs) || rs >= 0.05
            wtl = '=';
        elseif spkoa_mean < other_mean
            wtl = '+';
        else
            wtl = '-';
        end

        fprintf('  SPKOA vs %-8s : p = %.4e  [%s]\n', algorithms{alg_idx,2}, rs, wtl);
        rank_sum_resu = [rank_sum_resu, rs];
        wtl_resu      = [wtl_resu, {wtl}];
    end

    %% Friedman ranking data
    avg_performance  = mean(all_results, 2);
    rank_performance = tiedrank(avg_performance);
    RANK_DATA        = [RANK_DATA; rank_performance'];

    %% Store
    PVAL_RESULT{func_num}   = rank_sum_resu;
    WTLROW_RESULT{func_num} = wtl_resu;
    WTL_RESULT              = [WTL_RESULT; wtl_resu(1)];
    RESULT                  = [RESULT; resu];

    %% Convergence curves
    if plot_convergence == 1
        fig  = figure('Color', 'w', 'Position', [100 100 800 500]);
        k    = round(linspace(1, max_iter, CNT));
        iter = 1:max_iter;

        for alg_idx = 1:num_algorithms
            best_curve = all_curves{alg_idx, run+1};
            semilogy(iter(k), best_curve(k), ...
                [line_types{alg_idx}, markers{alg_idx}], ...
                'Color',      colors{alg_idx}, ...
                'LineWidth',  1.5, ...
                'MarkerSize', 6, ...
                'DisplayName', algorithms{alg_idx, 2});
            hold on;
        end

        grid on;
        title(sprintf('CEC2017 F%d (Dim=%d)', current_func, dim));
        xlabel('Iteration');
        ylabel('Best Fitness Value');
        box on;
        legend('Location', 'best');
        set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
        close(fig);
    end
end

%% Friedman ranking
friedman_mean = mean(RANK_DATA, 1);
[~, friedman_order] = sort(friedman_mean);
final_rank = zeros(1, num_algorithms);
for i = 1:length(friedman_order)
    final_rank(friedman_order(i)) = i;
end

fprintf('\n========== Friedman Ranking ==========\n');
for alg_idx = 1:num_algorithms
    fprintf('  %-10s Mean rank = %.4f  Final rank = %d\n', ...
        algorithms{alg_idx,2}, friedman_mean(alg_idx), final_rank(alg_idx));
end

%% Average runtime
avg_time_all = mean(TIME_RESULT, 1);
fprintf('\n========== Average Runtime ==========\n');
for alg_idx = 1:num_algorithms
    fprintf('  %-10s %.4f s\n', algorithms{alg_idx,2}, avg_time_all(alg_idx));
end

%% Win/Tie/Lose summary
fprintf('\n========== Win/Tie/Lose Summary ==========\n');
wins  = sum(strcmp(WTL_RESULT(:,1), '+'));
ties  = sum(strcmp(WTL_RESULT(:,1), '='));
loses = sum(strcmp(WTL_RESULT(:,1), '-'));
fprintf('  SPKOA vs KOA : +%d / =%d / -%d\n', wins, ties, loses);

%% Save to Excel
save_excel('SPKOA_vs_KOA_CEC2017.xlsx', RESULT, PVAL_RESULT, WTLROW_RESULT, ...
    F, algorithms(:,2), friedman_mean, final_rank, TIME_RESULT, avg_time_all);
fprintf('\nResults saved to SPKOA_vs_KOA_CEC2017.xlsx\n');

rmpath(genpath(pwd));


%% ========== Helper Functions ==========

function rs = perform_ranksum(z1, z2)
    z1 = z1(~isnan(z1));
    z2 = z2(~isnan(z2));
    if isempty(z1) || isempty(z2), rs = NaN; return; end
    if isequal(z1, z2), rs = 1; return; end
    [rs, ~] = ranksum(z1, z2);
    if isnan(rs), rs = NaN; return; end
    if rs < 3.0199e-1100, rs = 3.0199e-1100; end
end

function save_excel(filename, RESULT, PVAL_RESULT, WTLROW_RESULT, ...
        F, labels, friedman_mean, final_rank, TIME_RESULT, avg_time_all)

    num_rows_per_func = 7;
    num_funcs = length(F);
    num_algs  = length(labels);

    total_rows = num_funcs * num_rows_per_func + 4;
    A = cell(total_rows + 1, num_algs + 2);

    A(1,1) = {'Function'};
    A(1,2) = {'Metric'};
    A(1, 3:end) = labels';

    row_labels = {'Min'; 'Mean'; 'Std'; 'Max'; 'p-value'; '+/=/-'; 'Time(s)'};

    for i = 1:num_funcs
        base_result = (i-1) * 4;
        base_excel  = (i-1) * num_rows_per_func + 1;

        for k = 1:num_rows_per_func
            A{base_excel+k, 1} = sprintf('F%d', F(i));
            A{base_excel+k, 2} = row_labels{k};
        end

        for c = 1:num_algs
            for k = 1:4
                val = RESULT(base_result+k, c);
                A{base_excel+k, c+2} = val;
            end
        end

        p_row = PVAL_RESULT{i};
        for c = 1:num_algs
            val = p_row(c);
            if isnan(val)
                A{base_excel+5, c+2} = '--';
            else
                A{base_excel+5, c+2} = sprintf('%.4e', val);
            end
        end

        wtl_row = WTLROW_RESULT{i};
        for c = 1:num_algs
            A{base_excel+6, c+2} = wtl_row{c};
        end

        for c = 1:num_algs
            A{base_excel+7, c+2} = sprintf('%.4f', TIME_RESULT(i, c));
        end
    end

    friedman_base = num_funcs * num_rows_per_func + 2;
    A{friedman_base,   1} = 'Friedman';
    A{friedman_base,   2} = 'Mean rank';
    A{friedman_base+1, 1} = 'Friedman';
    A{friedman_base+1, 2} = 'Final rank';
    for c = 1:num_algs
        A{friedman_base,   c+2} = friedman_mean(c);
        A{friedman_base+1, c+2} = final_rank(c);
    end

    avg_time_base = num_funcs * num_rows_per_func + 4;
    A{avg_time_base, 1} = 'Average';
    A{avg_time_base, 2} = 'Time(s)';
    for c = 1:num_algs
        A{avg_time_base, c+2} = sprintf('%.4f', avg_time_all(c));
    end

    writecell(A, filename);
end