function spline3D_2()
    % Define the knot vectors for x and y
    knot_vectorx = [0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 32];
    knot_vectory = [0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 32];

    % Define the precision
    precision = 0.1;

    % Create mesh grids for x and y
    x = mesh(knot_vectorx(1), knot_vectorx(end), precision);
    y = mesh(knot_vectory(1), knot_vectory(end), precision);

    % Compute the spline coefficients for x and y
    px = compute_p(knot_vectorx);
    py = compute_p(knot_vectory);

    % Check the sanity of the knot vectors
    tx = check_sanity(knot_vectorx, px);
    ty = check_sanity(knot_vectory, py);

    % Calculate the number of basis functions
    nrx = compute_nr_basis_functions(knot_vectorx, px);
    nry = compute_nr_basis_functions(knot_vectory, py);

    % Initialize the coefficient matrix M
    M = zeros(length(x), length(y));

    % Initialize the weight matrix with random values
    weight_matrix = [
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
    0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1;
    1 1 0 1 1 1 0 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1;
    1 1 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0 1 0 1 1;
    1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1;
    1 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1;
    1 1 0 1 1 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 1;
    1 1 0 1 0 0 0 1 0 1 0 0 0 1 0 0 0 1 0 1 0 0 0 1 0 1 0 0 0 0 0 1 1;
    1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 1 1;
    1 1 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 1;
    1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 1 0 1 0 1 1;
    1 1 0 1 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 1;
    1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 1 1 0 1 1 1 1;
    1 1 0 0 0 0 0 1 0 1 0 0 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0 1 1;
    1 1 0 1 1 1 1 1 0 1 1 1 0 1 0 1 0 1 0 1 1 1 0 1 0 1 1 1 1 1 0 1 1;
    1 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 1 0 1 1;
    1 1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 0 1 1;
    1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 1 1;
    1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 0 1 0 1 1;
    1 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 1 0 0 0 0 0 1 0 1 1;
    1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 0 1 1;
    1 1 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 1 0 0 0 1 1;
    1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 0 1 0 1 1 1 1 1 1;
    1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 1 1;
    1 1 1 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 1;
    1 1 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 1 0 1 1;
    1 1 0 1 0 1 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 0 1 1;
    1 1 0 0 0 1 0 1 0 0 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1;
    1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 0 1 0 1 0 1 1;
    1 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 1 0 1 0 0 0;
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0;
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
   ]
    

    % Create the 3D mesh grid for plotting
    [X, Y] = meshgrid(x, y);

    % Compute the coefficient matrix M
    for i = 1:nrx
        for j = 1:nry
            spline1 = compute_spline(knot_vectorx, px, i, X) .* compute_spline(knot_vectory, py, j, Y);
            M = M + weight_matrix(i, j) * spline1;
        end
    end

    % Plot the 3D surface
    figure;
    surf(X, Y, M);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('3D Spline Surface');
end

function mesh_values = mesh(a, c, precision)
    mesh_values = a:precision:(c + precision);
end

function p = compute_p(knot_vector)
    initial = knot_vector(1);
    kvsize = length(knot_vector);
    i = 1;
    
    while (i + 1 < kvsize) && (initial == knot_vector(i + 1))
        i = i + 1;
    end
    
    p = i - 1;
end

function t = check_sanity(knot_vector, p)
    initial = knot_vector(1);
    kvsize = length(knot_vector);
    t = true;
    counter = 1;

    for i = 1:p + 1
        if (initial ~= knot_vector(i))
            t = false;
            return;
        end
    end

    for i = p + 2:kvsize - p - 1
        if (initial == knot_vector(i))
            counter = counter + 1;
            if (counter > p)
                t = false;
                return;
            end
        else
            initial = knot_vector(i);
            counter = 1;
        end
    end

    initial = knot_vector(kvsize);

    for i = kvsize - p:kvsize
        if (initial ~= knot_vector(i))
            t = false;
            return;
        end
    end

    for i = 1:kvsize - 1
        if (knot_vector(i) > knot_vector(i + 1))
            t = false;
        end
    end
end

function nr_basis = compute_nr_basis_functions(knot_vector, p)
    nr_basis = length(knot_vector) - p - 1;
end

function y = compute_spline(knot_vector, p, nr, x)
    fC = @(x, a, b) (x) / (b - a) - a / (b - a);
    fD = @(x, c, d) (1 - x) / (d - c) + (d - 1) / (d - c);

    a = knot_vector(nr);
    b = knot_vector(nr + p);
    c = knot_vector(nr + 1);
    d = knot_vector(nr + p + 1);

    if (p == 0)
        y = 0 .* (x < a) + 1 .* (a <= x & x <= d) + 0 .* (x > d);
        return;
    end

    lp = compute_spline(knot_vector, p - 1, nr, x);
    rp = compute_spline(knot_vector, p - 1, nr + 1, x);

    if (a == b)
        y1 = 0 .* (x < a) + 1 .* (a <= x & x <= b) + 0 .* (x > b);
    else
        y1 = 0 .* (x < a) + fC(x, a, b) .* (a <= x & x <= b) + 0 .* (x > b);
    end

    if (c == d)
        y2 = 0 .* (x < c) + 1 .* (c < x & x <= d) + 0 .* (d < x);
    else
        y2 = 0 .* (x < c) + fD(x, c, d) .* (c < x & x <= d) + 0 .* (d < x);
    end

    y = lp .* y1 + rp .* y2;
end