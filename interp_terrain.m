function zz = interp_terrain(xq, yq, x_grid, y_grid, z_grid)
    zz = interp2(x_grid, y_grid, z_grid, xq, yq, 'linear', 0);
end