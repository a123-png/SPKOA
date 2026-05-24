clc; clear; close all;
warning('off','MATLAB:exportgraphics:vectorWithTransparency');

%% ========== terrain ==========
[X, Y, Z, terrain_size] = generate_terrain_main();
x_grid = X; y_grid = Y; z_grid = Z;

%% ========== color ==========
alg_colors = {
    [0.85 0.33 0.10];
};


numParticles    = 50;
numMiddlePoints = 6;
dim  = numMiddlePoints * 3;
Tmax = 500;
nPop = round(numParticles);

margin = 50;
x_max  = max(x_grid(:));
y_max  = max(y_grid(:));

start_x = margin;
start_y = y_max - margin;
start_z = interp_terrain(start_x, start_y, x_grid, y_grid, z_grid) + 15;
startPoint = [start_x, start_y, start_z];

end_x = x_max - margin;
end_y = margin;
end_z = interp_terrain(end_x, end_y, x_grid, y_grid, z_grid) + 15;
endPoint = [end_x, end_y, end_z];

lb_way = [0,            0,            0 ];
ub_way = [terrain_size, terrain_size, 45];
lb = repmat(lb_way, 1, numMiddlePoints);
ub = repmat(ub_way, 1, numMiddlePoints);

%% ========== no_fly_zones ==========
no_fly_zones(1).cx     = 100;
no_fly_zones(1).cy     = 150;
no_fly_zones(1).radius = 25;
no_fly_zones(1).z_min  = 0;
no_fly_zones(1).z_max  = 45;

no_fly_zones(2).cx     = 180;
no_fly_zones(2).cy     = 80;
no_fly_zones(2).radius = 20;
no_fly_zones(2).z_min  = 0;
no_fly_zones(2).z_max  = 45;

%% ========== obstacles ==========
obstacles(1).pos0   = [165, 205, 25];
obstacles(1).vel    = [-3.68, -2.72, 0];
obstacles(1).radius = 12;
obstacles(1).type   = 'linear';
obstacles(1).x_lim  = [10, terrain_size-10];
obstacles(1).y_lim  = [10, terrain_size-10];
obstacles(1).z_lim  = [5, 45];
obstacles(1).t_end  = 25;

obstacles(2).pos0   = [40, 40, 25];
obstacles(2).vel    = [5.12, 2.56, 0];
obstacles(2).radius = 12;
obstacles(2).type   = 'linear';
obstacles(2).x_lim  = [10, terrain_size-10];
obstacles(2).y_lim  = [10, terrain_size-10];
obstacles(2).z_lim  = [5, 45];
obstacles(2).t_end  = 25;

t_per_segment = 0.08;

%% ========== fobj ==========
fobj.numMiddlePoints = numMiddlePoints;
fobj.startPoint      = startPoint;
fobj.endPoint        = endPoint;
fobj.terrain         = struct('x', x_grid, 'y', y_grid, 'z', z_grid);
fobj.lb              = lb_way;
fobj.ub              = ub_way;
fobj.terrain_z       = @(x,y) safe_interp2(x_grid, y_grid, z_grid, x, y);
fobj.no_fly_zones    = no_fly_zones;
fobj.obstacles       = obstacles;
fobj.t_per_segment   = t_per_segment;

%% ========== alg ==========
alg_names = {'SPKOA'};
alg_funcs = {
    @(n,t,u,l,d,f) SPKOA_optimization(n,t,u,l,d,f), ...
};
numAlgs = length(alg_names);
numRuns = 1;

all_scores   = zeros(numAlgs, numRuns);
all_times    = zeros(numAlgs, numRuns);
best_paths   = cell(numAlgs, 1);
best_curves  = cell(numAlgs, 1);
best_pos_all = cell(numAlgs, 1);

for i = 1:numAlgs
    fprintf('\nRunning algorithm: %s (%d runs)...\n', alg_names{i}, numRuns);
    best_score_i = inf;
    best_pos_i   = [];
    best_curve_i = [];
    for r = 1:numRuns
        tic;
        [score, pos, curve] = alg_funcs{i}(nPop, Tmax, ub, lb, dim, fobj);
        t_r = toc;
        all_scores(i,r) = score;
        all_times(i,r)  = t_r;
        if score < best_score_i
            best_score_i = score;
            best_pos_i   = pos;
            best_curve_i = curve;
        end
        fprintf('  Run %2d: fit=%.4f  time=%.1fs\n', r, score, t_r);
    end
    best_paths{i}   = generate_bspline_path(best_pos_i, startPoint, endPoint);
    best_curves{i}  = best_curve_i;
    best_pos_all{i} = best_pos_i;
    fprintf('%s done | best=%.4f\n', alg_names{i}, min(all_scores(i,:)));
end

%% ========== Statistics ==========
best_vals  = min(all_scores,  [], 2);
worst_vals = max(all_scores,  [], 2);
avg_vals   = mean(all_scores, 2);
std_vals   = std(all_scores,  0, 2);
avg_times  = mean(all_times,  2);

spkoa_idx = find(strcmp(alg_names,'SPKOA'));
p_values         = nan(numAlgs,1);
wilcoxon_results = cell(numAlgs,1);
for i = 1:numAlgs
    if i == spkoa_idx
        wilcoxon_results{i} = '--';
    else
        p = perform_ranksum(all_scores(spkoa_idx,:), all_scores(i,:));
        p_values(i) = p;
        if ~isnan(p) && p < 0.05
            if mean(all_scores(spkoa_idx,:)) < mean(all_scores(i,:))
                wilcoxon_results{i} = '+';
            else
                wilcoxon_results{i} = '-';
            end
        else
            wilcoxon_results{i} = '~';
        end
    end
end

fprintf('\n========== Statistics (%d runs) ==========\n', numRuns);
fprintf('%-8s %-12s %-12s %-12s %-10s %-12s %-10s %-10s\n', ...
    'Algorithm','Best','Worst','Mean','Std','AvgTime(s)','P_Value','Wilcoxon');
fprintf('%s\n', repmat('-',1,88));
for i = 1:numAlgs
    if isnan(p_values(i))
        fprintf('%-8s %-12.4f %-12.4f %-12.4f %-10.4f %-12.2f %-10s %-10s\n', ...
            alg_names{i},best_vals(i),worst_vals(i),avg_vals(i),std_vals(i),avg_times(i),'NaN',wilcoxon_results{i});
    else
        fprintf('%-8s %-12.4f %-12.4f %-12.4f %-10.4f %-12.2f %-10.4f %-10s\n', ...
            alg_names{i},best_vals(i),worst_vals(i),avg_vals(i),std_vals(i),avg_times(i),p_values(i),wilcoxon_results{i});
    end
end

%% ========== Path Quality Metrics ==========
best_path_lengths     = zeros(numAlgs,1);
max_yaw_angles        = zeros(numAlgs,1);
max_pitch_angles      = zeros(numAlgs,1);
altitude_variation    = zeros(numAlgs,1);
smoothness_vals       = zeros(numAlgs,1);
constraint_viols      = zeros(numAlgs,1);
min_terrain_clearance = zeros(numAlgs,1);

for i = 1:numAlgs
    path_i = best_paths{i};
    if isempty(path_i), continue; end

    best_path_lengths(i) = computePathLength(path_i);

    best_pos_i = best_pos_all{i};
    if ~isempty(best_pos_i)
        pts_ctrl      = reshape(best_pos_i, 3, fobj.numMiddlePoints)';
        pts_full_ctrl = [startPoint; pts_ctrl; endPoint];
        dp_ctrl       = diff(pts_full_ctrl, 1, 1);
        dx_ctrl = dp_ctrl(:,1); dy_ctrl = dp_ctrl(:,2); dz_ctrl = dp_ctrl(:,3);

        vec1_ctrl = [dx_ctrl(1:end-1), dy_ctrl(1:end-1)];
        vec2_ctrl = [dx_ctrl(2:end),   dy_ctrl(2:end)  ];
        n1_ctrl = sqrt(sum(vec1_ctrl.^2,2)); n2_ctrl = sqrt(sum(vec2_ctrl.^2,2));
        valid_v_ctrl = (n1_ctrl>1e-6) & (n2_ctrl>1e-6);
        cos_theta_ctrl = sum(vec1_ctrl(valid_v_ctrl,:).*vec2_ctrl(valid_v_ctrl,:),2) ...
                         ./(n1_ctrl(valid_v_ctrl).*n2_ctrl(valid_v_ctrl));
        cos_theta_ctrl  = min(1,max(-1,cos_theta_ctrl));
        yaw_angles_ctrl = abs(acos(cos_theta_ctrl));
        max_yaw_angles(i)  = rad2deg(max(yaw_angles_ctrl));
        smoothness_vals(i) = rad2deg(sqrt(mean(yaw_angles_ctrl.^2)));
        viol_yaw = sum(yaw_angles_ctrl > deg2rad(45));

        horiz_ctrl   = sqrt(dx_ctrl.^2 + dy_ctrl.^2);
        valid_h_ctrl = horiz_ctrl > 1e-6;
        pitch_angles = abs(atan2(dz_ctrl(valid_h_ctrl), horiz_ctrl(valid_h_ctrl)));
        max_pitch_angles(i) = rad2deg(max(pitch_angles));
        viol_pitch = sum(pitch_angles > deg2rad(45));
    else
        viol_yaw = 0; viol_pitch = 0;
    end

    altitude_variation(i) = max(path_i(:,3)) - min(path_i(:,3));
    constraint_viols(i)   = viol_yaw + viol_pitch;

    clearances = zeros(size(path_i,1), 1);
    for k = 1:size(path_i,1)
        terrain_z_k  = safe_interp2(x_grid, y_grid, z_grid, path_i(k,1), path_i(k,2));
        clearances(k) = path_i(k,3) - terrain_z_k;
    end
    min_terrain_clearance(i) = min(clearances);
end

fprintf('\n========== Path Quality Metrics ==========\n');
fprintf('%-8s %-15s %-15s %-20s %-15s %-22s %-20s\n', ...
    'Algorithm','MaxYaw(deg)','MaxPitch(deg)','AltVariation(m)','Smoothness','ConstraintViols','MinClearance(m)');
fprintf('%s\n', repmat('-',1,115));
for i = 1:numAlgs
    fprintf('%-8s %-15.3f %-15.3f %-20.3f %-15.6f %-22d %-20.3f\n', ...
        alg_names{i}, max_yaw_angles(i), max_pitch_angles(i), ...
        altitude_variation(i), smoothness_vals(i), constraint_viols(i), ...
        min_terrain_clearance(i));
end

%% ========== Convergence Curve ==========
figure('Color','w','Position',[100 100 800 550]);
hold on;
for i = 1:numAlgs
    curve_i = best_curves{i};
    if isempty(curve_i), continue; end
    curve_i = curve_i(:)';
    curve_i = min(curve_i, 500);
    plot(1:length(curve_i), curve_i, '-', 'LineWidth', 1.8, 'Color', alg_colors{i});
end
xlabel('Iteration', 'FontSize', 13);
ylabel('Cost', 'FontSize', 13);
legend(alg_names, 'Location', 'northeast', 'FontSize', 10);
box on; grid on;
set(gca, 'FontSize', 12);
hold off;

%% ========== Path Window ==========
for alg_i = 1:numAlgs
    show_single_path(x_grid, y_grid, z_grid, best_paths{alg_i}, ...
        alg_names{alg_i}, alg_colors{alg_i}, alg_i, ...
        no_fly_zones, obstacles, t_per_segment, startPoint, endPoint);
end

%% ========== Animation Window ==========
fprintf('\nGenerating animation windows...\n');
uav_speed = 10;
for alg_i = 1:numAlgs
    animate_flight(x_grid, y_grid, z_grid, best_paths{alg_i}, obstacles, ...
        no_fly_zones, startPoint, endPoint, t_per_segment, uav_speed, ...
        alg_names{alg_i}, alg_colors{alg_i});

    show_interactive_result(x_grid, y_grid, z_grid, ...
        best_paths{alg_i}, best_curves{alg_i}, startPoint, endPoint, ...
        no_fly_zones, obstacles, t_per_segment, ...
        min(all_scores(alg_i,:)), computePathLength(best_paths{alg_i}), ...
        avg_times(alg_i), alg_names{alg_i}, alg_colors{alg_i});
end

disp('All done!');


%% ===================================================================
%%  Local function: interactive result window
%% ===================================================================
function fig = show_interactive_result(x_grid, y_grid, z_grid, best_path, best_curve, ...
    startPoint, endPoint, no_fly_zones, obstacles, t_per_segment, ...
    best_score, path_len, t_elapsed, alg_name, path_color)

    if nargin < 15, path_color = [0.85 0.33 0.10]; end

    fig = figure('Name',sprintf('%s Path Planning',alg_name), ...
        'NumberTitle','off','Color',[0.15 0.15 0.15],'Position',[80 60 1300 750]);

    uicontrol('Style','text','String',sprintf('%s Path Planning Result', alg_name), ...
        'Units','normalized','Position',[0.0 0.93 1.0 0.07], ...
        'FontSize',16,'FontWeight','bold','ForegroundColor','w','BackgroundColor',[0.15 0.15 0.15]);

    ax3d = axes('Parent',fig,'Units','normalized','Position',[0.02 0.18 0.55 0.72], ...
        'Color',[0.1 0.1 0.1],'GridColor',[0.4 0.4 0.4],'XColor','w','YColor','w','ZColor','w','FontSize',10);
    hold(ax3d,'on');
    surf(ax3d,x_grid,y_grid,z_grid,'EdgeColor','none','FaceAlpha',0.72);
    colormap(ax3d,parula);

    theta_c = linspace(0,2*pi,60);
    for j=1:length(no_fly_zones)
        nfz=no_fly_zones(j);
        xc=nfz.cx+nfz.radius*cos(theta_c);
        yc=nfz.cy+nfz.radius*sin(theta_c);
        zs=linspace(nfz.z_min,nfz.z_max,20);
        [Tc,Zc]=meshgrid(theta_c,zs);
        surf(ax3d,nfz.cx+nfz.radius*cos(Tc),nfz.cy+nfz.radius*sin(Tc),Zc,...
            'FaceColor',[1 0.15 0.15],'FaceAlpha',0.22,'EdgeColor','none');
        plot3(ax3d,xc,yc,ones(size(xc))*nfz.z_max,'r-','LineWidth',2);
        plot3(ax3d,xc,yc,ones(size(xc))*nfz.z_min,'r-','LineWidth',2);
    end

    n_shots=6; obs_color=[1.0,0.55,0.0]; [sx,sy,sz]=sphere(24);
    t_steps=linspace(0,size(best_path,1)*t_per_segment,n_shots);
    alphas=linspace(0.08,0.50,n_shots);
    for j=1:length(obstacles)
        traj=zeros(n_shots,3);
        for ti=1:n_shots
            op=get_obstacle_pos(obstacles(j),t_steps(ti));
            traj(ti,:)=op; r=obstacles(j).radius;
            surf(ax3d,sx*r+op(1),sy*r+op(2),sz*r+op(3),...
                'FaceColor',obs_color,'FaceAlpha',alphas(ti),'EdgeColor','none');
        end
        plot3(ax3d,traj(:,1),traj(:,2),traj(:,3),'--','Color',obs_color,'LineWidth',1.8);
    end

    plot3(ax3d,best_path(:,1),best_path(:,2),best_path(:,3),'-','LineWidth',2.8,'Color',path_color);
    scatter3(ax3d,startPoint(1),startPoint(2),startPoint(3),180,'g','^','filled');
    scatter3(ax3d,endPoint(1),  endPoint(2),  endPoint(3),  200,'r','p','filled');
    xlabel(ax3d,'X','Color','w','FontSize',11);
    ylabel(ax3d,'Y','Color','w','FontSize',11);
    zlabel(ax3d,'Z','Color','w','FontSize',11);
    title(ax3d,'','Color','w','FontSize',12);
    grid(ax3d,'on'); box(ax3d,'on'); view(ax3d,145,35); rotate3d(ax3d,'on');
    hold(ax3d,'off');

    ax2d = axes('Parent',fig,'Units','normalized','Position',[0.60 0.46 0.38 0.44], ...
        'Color',[0.1 0.1 0.1],'XColor','w','YColor','w','FontSize',10);
    hold(ax2d,'on');
    imagesc(ax2d,x_grid(1,:),y_grid(:,1),z_grid);
    colormap(ax2d,parula); set(ax2d,'YDir','normal'); alpha(ax2d,0.75);
    for j=1:length(no_fly_zones)
        nfz=no_fly_zones(j);
        fill(ax2d,nfz.cx+nfz.radius*cos(theta_c),nfz.cy+nfz.radius*sin(theta_c),...
            'r','FaceAlpha',0.28,'EdgeColor','r','LineWidth',2);
    end
    line_styles={'-','--','-','--','-','--'};
    alphas2d=linspace(0.10,0.45,n_shots);
    for j=1:length(obstacles)
        traj2=zeros(n_shots,2);
        for ti=1:n_shots
            op=get_obstacle_pos(obstacles(j),t_steps(ti));
            traj2(ti,:)=op(1:2); r=obstacles(j).radius;
            xo=op(1)+r*cos(theta_c); yo=op(2)+r*sin(theta_c);
            fill(ax2d,xo,yo,obs_color,'FaceAlpha',alphas2d(ti),...
                'EdgeColor',obs_color,'LineWidth',1.5,'LineStyle',line_styles{ti});
        end
        plot(ax2d,traj2(:,1),traj2(:,2),'--','Color',obs_color,'LineWidth',1.5);
        if n_shots>=2
            dp=traj2(end,:)-traj2(end-1,:);
            quiver(ax2d,traj2(end-1,1),traj2(end-1,2),dp(1),dp(2),0,...
                'Color',obs_color,'LineWidth',2,'MaxHeadSize',3);
        end
    end
    plot(ax2d,best_path(:,1),best_path(:,2),'-','LineWidth',2.2,'Color',path_color);
    scatter(ax2d,startPoint(1),startPoint(2),120,'g','^','filled');
    scatter(ax2d,endPoint(1),  endPoint(2),  120,'r','p','filled');
    xlabel(ax2d,'X','Color','w','FontSize',10); ylabel(ax2d,'Y','Color','w','FontSize',10);
    title(ax2d,'','Color','w','FontSize',11);
    axis(ax2d,'tight'); grid(ax2d,'on'); box(ax2d,'on');
    hold(ax2d,'off');

    ax_cv = axes('Parent',fig,'Units','normalized','Position',[0.60 0.05 0.38 0.33], ...
        'Color',[0.1 0.1 0.1],'XColor','w','YColor','w','FontSize',10);
    hold(ax_cv,'on');
    if ~isempty(best_curve)
        curve_row=best_curve(:)'; iters=1:length(curve_row);
        fill(ax_cv,[iters,fliplr(iters)],[curve_row,ones(1,length(iters))*max(curve_row)],...
            path_color,'FaceAlpha',0.15,'EdgeColor','none');
        plot(ax_cv,iters,curve_row,'-','LineWidth',2,'Color',path_color);
    end
    xlabel(ax_cv,'Iteration','Color','w','FontSize',10);
    ylabel(ax_cv,'Cost','Color','w','FontSize',10);
    title(ax_cv,'','Color','w','FontSize',11);
    grid(ax_cv,'on'); box(ax_cv,'on');
    hold(ax_cv,'off');

    info_str = sprintf('  Cost: %.4f   PathLen: %.2f m   Time: %.1f s   NFZ: %d   OBS: %d', ...
        best_score, path_len, t_elapsed, length(no_fly_zones), length(obstacles));
    uicontrol('Style','text','String',info_str,'Units','normalized','Position',[0.0 0.0 1.0 0.055], ...
        'FontSize',12,'FontWeight','bold','ForegroundColor',[0.2 1.0 0.4],...
        'BackgroundColor',[0.1 0.1 0.1],'HorizontalAlignment','left');

    btn_cfg={'Units','normalized','FontSize',10,'FontWeight','bold','ForegroundColor','w'};
    uicontrol('Style','pushbutton','String','Front',btn_cfg{:},'Position',[0.02 0.06 0.07 0.045],...
        'BackgroundColor',[0.25 0.25 0.35],'Callback',@(~,~) view(ax3d,145,35));
    uicontrol('Style','pushbutton','String','Top',btn_cfg{:},'Position',[0.10 0.06 0.07 0.045],...
        'BackgroundColor',[0.25 0.25 0.35],'Callback',@(~,~) view(ax3d,0,90));
    uicontrol('Style','pushbutton','String','Side',btn_cfg{:},'Position',[0.18 0.06 0.07 0.045],...
        'BackgroundColor',[0.25 0.25 0.35],'Callback',@(~,~) view(ax3d,0,0));
end


%% ===================================================================
%%  Local function: flight animation
%% ===================================================================
function fig = animate_flight(x_grid, y_grid, z_grid, path, obstacles, ...
    no_fly_zones, startPoint, endPoint, t_per_segment, uav_speed, alg_name, path_color)

    if nargin < 12, path_color = [0.85 0.33 0.10]; end

    n_pts=size(path,1);
    seg_len=sqrt(sum(diff(path).^2,2));
    cum_dist=[0;cumsum(seg_len)];
    t_uav=cum_dist/uav_speed;
    total_time=t_uav(end);
    n_frames=120;
    t_frames=linspace(0,total_time,n_frames);

    dist_all=zeros(n_pts,length(obstacles));
    for k=1:n_pts
        for j=1:length(obstacles)
            op=get_obstacle_pos(obstacles(j),t_uav(k));
            dist_all(k,j)=norm(path(k,:)-op);
        end
    end

    fig=figure('Name',sprintf('%s Animation',alg_name),'Color',[0.12 0.12 0.12],'Position',[60 60 1100 680]);
    theta_c=linspace(0,2*pi,60); [sx,sy,sz]=sphere(20);

    ax3d=axes('Parent',fig,'Units','normalized','Position',[0.02 0.13 0.60 0.83],...
        'Color',[0.08 0.08 0.08],'XColor','w','YColor','w','ZColor','w','GridColor',[0.3 0.3 0.3],'FontSize',10);
    hold(ax3d,'on');
    surf(ax3d,x_grid,y_grid,z_grid,'EdgeColor','none','FaceAlpha',0.6); colormap(ax3d,parula);
    for j=1:length(no_fly_zones)
        nfz=no_fly_zones(j);
        xc=nfz.cx+nfz.radius*cos(theta_c); yc=nfz.cy+nfz.radius*sin(theta_c);
        zs=linspace(nfz.z_min,nfz.z_max,15); [Tc,Zc]=meshgrid(theta_c,zs);
        surf(ax3d,nfz.cx+nfz.radius*cos(Tc),nfz.cy+nfz.radius*sin(Tc),Zc,...
            'FaceColor',[1 0.1 0.1],'FaceAlpha',0.18,'EdgeColor','none');
        plot3(ax3d,xc,yc,ones(size(xc))*nfz.z_max,'r-','LineWidth',1.8);
        plot3(ax3d,xc,yc,ones(size(xc))*nfz.z_min,'r-','LineWidth',1.8);
    end
    plot3(ax3d,path(:,1),path(:,2),path(:,3),'--','Color',[0.5 0.5 0.5],'LineWidth',1);
    scatter3(ax3d,startPoint(1),startPoint(2),startPoint(3),160,'g','^','filled');
    scatter3(ax3d,endPoint(1),endPoint(2),endPoint(3),180,'r','p','filled');
    h_flown=plot3(ax3d,NaN,NaN,NaN,'-','Color',path_color,'LineWidth',2.5);
    h_uav=scatter3(ax3d,NaN,NaN,NaN,250,'w','filled','o');
    h_obs3d=cell(length(obstacles),1);
    for j=1:length(obstacles)
        op=get_obstacle_pos(obstacles(j),0); r=obstacles(j).radius;
        h_obs3d{j}=surf(ax3d,sx*r+op(1),sy*r+op(2),sz*r+op(3),...
            'FaceColor',[1 0.55 0],'FaceAlpha',0.55,'EdgeColor','none');
    end
    xlabel(ax3d,'X','Color','w'); ylabel(ax3d,'Y','Color','w'); zlabel(ax3d,'Z','Color','w');
    grid(ax3d,'on'); box(ax3d,'on'); view(ax3d,145,35);
    hold(ax3d,'off');

    ax2d=axes('Parent',fig,'Units','normalized','Position',[0.65 0.42 0.33 0.54],...
        'Color',[0.08 0.08 0.08],'XColor','w','YColor','w','FontSize',9);
    hold(ax2d,'on');
    imagesc(ax2d,x_grid(1,:),y_grid(:,1),z_grid);
    colormap(ax2d,parula); set(ax2d,'YDir','normal'); alpha(ax2d,0.65);
    for j=1:length(no_fly_zones)
        nfz=no_fly_zones(j);
        fill(ax2d,nfz.cx+nfz.radius*cos(theta_c),nfz.cy+nfz.radius*sin(theta_c),...
            'r','FaceAlpha',0.22,'EdgeColor','r','LineWidth',1.5);
    end
    plot(ax2d,path(:,1),path(:,2),'--','Color',[0.5 0.5 0.5],'LineWidth',1);
    scatter(ax2d,startPoint(1),startPoint(2),80,'g','^','filled');
    scatter(ax2d,endPoint(1),endPoint(2),80,'r','p','filled');
    h_flown2d=plot(ax2d,NaN,NaN,'-','Color',path_color,'LineWidth',2);
    h_uav2d=scatter(ax2d,NaN,NaN,140,'w','filled','o');
    h_obs2d=cell(length(obstacles),1);
    for j=1:length(obstacles)
        h_obs2d{j}=fill(ax2d,NaN,NaN,[1 0.55 0],'FaceAlpha',0.45,'EdgeColor',[1 0.55 0],'LineWidth',1.5);
    end
    title(ax2d,'Top View','Color','w','FontSize',10);
    axis(ax2d,'tight'); grid(ax2d,'on'); box(ax2d,'on'); hold(ax2d,'off');

    obs_colors={[1 0.55 0],[0.3 0.75 1],[0.5 1 0.3]};
    ax_dist=axes('Parent',fig,'Units','normalized','Position',[0.65 0.05 0.33 0.30],...
        'Color',[0.08 0.08 0.08],'XColor','w','YColor','w','FontSize',9);
    hold(ax_dist,'on');
    h_xline=cell(length(obstacles),1);
    for j=1:length(obstacles)
        plot(ax_dist,t_uav,dist_all(:,j),'-','Color',obs_colors{mod(j-1,3)+1},'LineWidth',1.5);
        yline(ax_dist,obstacles(j).radius,'--','Color',obs_colors{mod(j-1,3)+1},'LineWidth',1.5);
        h_xline{j}=xline(ax_dist,0,'w-','LineWidth',2);
    end
    xlabel(ax_dist,'Time (s)','Color','w','FontSize',9);
    ylabel(ax_dist,'Distance (m)','Color','w','FontSize',9);
    title(ax_dist,'Real-time Distance','Color','w','FontSize',10);
    xlim(ax_dist,[0 total_time]); ylim(ax_dist,[0 max(dist_all(:))*1.1]);
    grid(ax_dist,'on'); box(ax_dist,'on'); hold(ax_dist,'off');

    h_info=uicontrol('Parent',fig,'Style','text','Units','normalized',...
        'Position',[0.02 0.005 0.56 0.04],'FontSize',10,'FontWeight','bold',...
        'ForegroundColor',[0.2 1 0.5],'BackgroundColor',[0.08 0.08 0.08],...
        'HorizontalAlignment','left','String','Ready...');

    btn_w=0.09; btn_h=0.045; btn_y=0.005;
    uicontrol('Parent',fig,'Style','pushbutton','String','Play','Units','normalized',...
        'Position',[0.60 btn_y btn_w btn_h],'FontSize',10,'FontWeight','bold','ForegroundColor','w',...
        'BackgroundColor',[0.20 0.45 0.20],'Callback',@(~,~) start_anim());
    uicontrol('Parent',fig,'Style','pushbutton','String','Pause','Units','normalized',...
        'Position',[0.70 btn_y btn_w btn_h],'FontSize',10,'FontWeight','bold','ForegroundColor','w',...
        'BackgroundColor',[0.25 0.25 0.40],'Callback',@(~,~) pause_anim());
    uicontrol('Parent',fig,'Style','pushbutton','String','Stop','Units','normalized',...
        'Position',[0.80 btn_y btn_w btn_h],'FontSize',10,'FontWeight','bold','ForegroundColor','w',...
        'BackgroundColor',[0.40 0.20 0.20],'Callback',@(~,~) stop_anim());

    uicontrol('Parent',fig,'Style','text','String','View:','Units','normalized',...
        'Position',[0.02 btn_y+btn_h+0.005 0.05 0.03],'ForegroundColor',[0.8 0.8 0.8],...
        'BackgroundColor',[0.08 0.08 0.08],'FontSize',9);
    uicontrol('Parent',fig,'Style','pushbutton','String','Front','Units','normalized',...
        'Position',[0.08 btn_y+btn_h+0.005 0.06 0.03],'ForegroundColor','w',...
        'BackgroundColor',[0.20 0.20 0.30],'FontSize',9,'Callback',@(~,~) view(ax3d,145,35));
    uicontrol('Parent',fig,'Style','pushbutton','String','Top','Units','normalized',...
        'Position',[0.15 btn_y+btn_h+0.005 0.06 0.03],'ForegroundColor','w',...
        'BackgroundColor',[0.20 0.20 0.30],'FontSize',9,'Callback',@(~,~) view(ax3d,0,90));
    uicontrol('Parent',fig,'Style','pushbutton','String','Side','Units','normalized',...
        'Position',[0.22 btn_y+btn_h+0.005 0.06 0.03],'ForegroundColor','w',...
        'BackgroundColor',[0.20 0.20 0.30],'FontSize',9,'Callback',@(~,~) view(ax3d,0,0));

    state.frame=1; state.saving=false;
    state.gif_file=sprintf('%s_flight_animation.gif',alg_name);

    tmr=timer('ExecutionMode','fixedRate','Period',0.05,...
        'TimerFcn',@update_frame,'StopFcn',@on_stop);

    function start_anim()
        if ~ishandle(fig), return; end
        if strcmp(tmr.Running,'off'), state.frame=1; start(tmr); end
    end
    function pause_anim()
        if ~ishandle(fig), return; end
        if strcmp(tmr.Running,'on'), stop(tmr);
        elseif state.frame<=n_frames, start(tmr); end
    end
    function stop_anim()
        if strcmp(tmr.Running,'on'), stop(tmr); end
        state.frame=1; set(h_info,'String','Stopped. Press Play to restart.');
    end
    function on_stop(~,~)
    end
    function update_frame(~,~)
        if ~ishandle(fig), stop(tmr); return; end
        if state.frame>n_frames
            stop(tmr); set(h_info,'String',sprintf('Finished. Total: %.1fs',total_time)); return;
        end
        t_now=t_frames(state.frame);
        ux=interp1(t_uav,path(:,1),t_now); uy=interp1(t_uav,path(:,2),t_now); uz=interp1(t_uav,path(:,3),t_now);
        mask=t_uav<=t_now; fp=path(mask,:);
        if size(fp,1)>=1, set(h_flown,'XData',fp(:,1),'YData',fp(:,2),'ZData',fp(:,3)); end
        set(h_uav,'XData',ux,'YData',uy,'ZData',uz);
        for jj=1:length(obstacles)
            op=get_obstacle_pos(obstacles(jj),t_now); r=obstacles(jj).radius;
            set(h_obs3d{jj},'XData',sx*r+op(1),'YData',sy*r+op(2),'ZData',sz*r+op(3));
        end
        if size(fp,1)>=1, set(h_flown2d,'XData',fp(:,1),'YData',fp(:,2)); end
        set(h_uav2d,'XData',ux,'YData',uy);
        for jj=1:length(obstacles)
            op=get_obstacle_pos(obstacles(jj),t_now); r=obstacles(jj).radius;
            xo=op(1)+r*cos(theta_c); yo=op(2)+r*sin(theta_c);
            set(h_obs2d{jj},'XData',xo,'YData',yo);
        end
        for jj=1:length(obstacles), set(h_xline{jj},'Value',t_now); end
        dist_str='';
        for jj=1:length(obstacles)
            op=get_obstacle_pos(obstacles(jj),t_now); d=norm([ux,uy,uz]-op); r=obstacles(jj).radius;
            if d<r, st='!Collision'; elseif d<r*1.5, st='!Warning'; else, st='Safe'; end
            dist_str=[dist_str,sprintf(' OBS%d:%.1fm%s',jj,d,st)]; %#ok
        end
        pct=state.frame/n_frames*100;
        set(h_info,'String',sprintf(' t=%.1f/%.1fs (%.0f%%) UAV(%.0f,%.0f,%.0f)%s',...
            t_now,total_time,pct,ux,uy,uz,dist_str));
        drawnow;
        state.frame=state.frame+1;
    end

    set(fig,'CloseRequestFcn','closereq');
    addlistener(fig,'ObjectBeingDestroyed',@(~,~) safe_stop(tmr));
    fprintf('Animation window opened. Press Play to start.\n');
end

function safe_stop(tmr)
    try
        if isvalid(tmr) && strcmp(tmr.Running,'on'), stop(tmr); end
        if isvalid(tmr), delete(tmr); end
    catch, end
end


%% ===================================================================
%%  Local function: single path window
%% ===================================================================
function fig = show_single_path(x_grid, y_grid, z_grid, path, alg_name, path_color, alg_idx, ...
    no_fly_zones, obstacles, t_per_segment, startPoint, endPoint)

    theta_c=linspace(0,2*pi,60);
    n_shots=6; obs_color=[1.0,0.55,0.0]; [sx,sy,sz]=sphere(20);
    offset=(alg_idx-1)*30;

    fig=figure('Name',sprintf('%s Path',alg_name),'Color',[0.13 0.13 0.13],...
        'Position',[50+offset 50+offset 950 700]);

    ax=axes('Parent',fig,'Units','normalized','Position',[0.05 0.17 0.88 0.78],...
        'Color',[0.08 0.08 0.08],'XColor','w','YColor','w','ZColor','w',...
        'GridColor',[0.3 0.3 0.3],'FontSize',10);
    hold(ax,'on');
    surf(ax,x_grid,y_grid,z_grid,'EdgeColor','none','FaceAlpha',0.62); colormap(ax,parula);
    for j=1:length(no_fly_zones)
        nfz=no_fly_zones(j); xc=nfz.cx+nfz.radius*cos(theta_c); yc=nfz.cy+nfz.radius*sin(theta_c);
        zs=linspace(nfz.z_min,nfz.z_max,15); [Tc,Zc]=meshgrid(theta_c,zs);
        surf(ax,nfz.cx+nfz.radius*cos(Tc),nfz.cy+nfz.radius*sin(Tc),Zc,...
            'FaceColor','r','FaceAlpha',0.2,'EdgeColor','none');
        plot3(ax,xc,yc,ones(size(xc))*nfz.z_max,'r-','LineWidth',1.8);
    end
    n_seg=size(path,1);
    t_steps=linspace(0,n_seg*t_per_segment,n_shots);
    alphas=linspace(0.08,0.50,n_shots);
    for j=1:length(obstacles)
        traj=zeros(n_shots,3);
        for ti=1:n_shots
            op=get_obstacle_pos(obstacles(j),t_steps(ti)); traj(ti,:)=op; r=obstacles(j).radius;
            surf(ax,sx*r+op(1),sy*r+op(2),sz*r+op(3),...
                'FaceColor',obs_color,'FaceAlpha',alphas(ti),'EdgeColor','none');
        end
        plot3(ax,traj(:,1),traj(:,2),traj(:,3),'--','Color',obs_color,'LineWidth',1.5);
    end
    plot3(ax,path(:,1),path(:,2),path(:,3),'-','LineWidth',2.8,'Color',path_color);
    scatter3(ax,startPoint(1),startPoint(2),startPoint(3),140,'g','^','filled');
    scatter3(ax,endPoint(1),  endPoint(2),  endPoint(3),  160,'r','p','filled');
    xlabel(ax,'X','Color','w','FontSize',11); ylabel(ax,'Y','Color','w','FontSize',11);
    zlabel(ax,'Z','Color','w','FontSize',11);
    title(ax,'','Color','w','FontSize',13);
    grid(ax,'on'); box(ax,'on'); axis(ax,'tight'); view(ax,145,35); rotate3d(ax,'on');
    hold(ax,'off');

    btn_h=0.05; btn_y=0.01;
    btn_cfg={'Units','normalized','FontSize',10,'FontWeight','bold','ForegroundColor','w'};
    uicontrol('Parent',fig,'Style','pushbutton','String','Front',btn_cfg{:},...
        'Position',[0.02 btn_y 0.09 btn_h],'BackgroundColor',[0.22 0.22 0.35],'Callback',@(~,~) view(ax,145,35));
    uicontrol('Parent',fig,'Style','pushbutton','String','Top(XY)',btn_cfg{:},...
        'Position',[0.12 btn_y 0.09 btn_h],'BackgroundColor',[0.22 0.22 0.35],'Callback',@(~,~) view(ax,0,90));
    uicontrol('Parent',fig,'Style','pushbutton','String','Side(YZ)',btn_cfg{:},...
        'Position',[0.22 btn_y 0.09 btn_h],'BackgroundColor',[0.22 0.22 0.35],'Callback',@(~,~) view(ax,90,0));
    uicontrol('Parent',fig,'Style','pushbutton','String','Side(XZ)',btn_cfg{:},...
        'Position',[0.32 btn_y 0.09 btn_h],'BackgroundColor',[0.22 0.22 0.35],'Callback',@(~,~) view(ax,0,0));
end


%% ===================================================================
%%  Local function: Wilcoxon rank-sum test
%% ===================================================================
function p = perform_ranksum(z1, z2)
    z1 = z1(~isnan(z1));
    z2 = z2(~isnan(z2));
    if isempty(z1) || isempty(z2)
        p = NaN; return;
    end
    if isequal(z1, z2)
        p = 1; return;
    end
    try
        [p, ~] = ranksum(z1, z2);
    catch
        p = NaN; return;
    end
    if isnan(p)
        p = NaN; return;
    end
    p_min = 3.0199e-1100;
    if p < p_min
        p = p_min;
    end
end