function [X, Y, Z, terrain_size] = generate_terrain_main()
    rng(32);  
    terrain_size = 250; 
    n_peaks = 15; 
    
    peaks = generate_peaks(n_peaks, terrain_size); 
    x = linspace(0, terrain_size, 120); 
    y = linspace(0, terrain_size, 120);
    [X, Y] = meshgrid(x, y);
    Z = terrain_model(X, Y, peaks, terrain_size); 

      
end


function peaks = generate_peaks(n_peaks, terrain_size)
    peaks = zeros(n_peaks, 5);
    width_base = terrain_size / 25; 
    height_base = 15; 
    
    for i = 1:n_peaks
        x_i = rand * terrain_size;   
        y_i = rand * terrain_size;   
        h_i = height_base + rand * 30; 
        x_si = width_base + rand * width_base*2; 
        y_si = width_base + rand * width_base*2; 
        peaks(i, :) = [x_i, y_i, h_i, x_si, y_si];
    end
end

function Z = terrain_model(X, Y, peaks, terrain_size)
    Z = zeros(size(X)); 
    n = size(peaks, 1); 
    
    for i = 1:n
        x_i = peaks(i, 1);
        y_i = peaks(i, 2);
        h_i = peaks(i, 3);
        x_si = peaks(i, 4);
        y_si = peaks(i, 5);
        

        r_sq = ((X - x_i)/x_si).^2 + ((Y - y_i)/y_si).^2;
        single_peak = h_i * cos( (sqrt(r_sq) * pi)/2 ) .* (r_sq <= 1);
        

        boundary_mask = (X>0) .* (X<terrain_size) .* (Y>0) .* (Y<terrain_size);
        single_peak = single_peak .* boundary_mask;
        
        Z = Z + single_peak;
    end
end