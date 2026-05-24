function path = generate_bspline_path(midpoints, startPoint, endPoint)
    pts_full = [startPoint; reshape(midpoints, 3, [])'; endPoint];
    t = linspace(0, 1, size(pts_full,1));
    tt = linspace(0, 1, 500); 
    path_x = spline(t, pts_full(:,1), tt);
    path_y = spline(t, pts_full(:,2), tt);
    path_z = spline(t, pts_full(:,3), tt);
    path = [path_x', path_y', path_z'];
end
