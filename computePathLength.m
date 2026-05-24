function len = computePathLength(path)
    diffPath = diff(path, 1, 1);
    len = sum(sqrt(sum(diffPath.^2, 2)));
end