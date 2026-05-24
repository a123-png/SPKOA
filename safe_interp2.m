function z = safe_interp2(X, Y, Z, x, y)

    x(~isfinite(x)) = NaN;
    y(~isfinite(y)) = NaN;
    try
        z = interp2(X, Y, Z, x, y, 'linear', NaN);
    catch
        z = NaN(size(x));  
    end
end
