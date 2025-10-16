function bspline_curve(precision, knot_vector, points)
    if mod(precision, 2) == 0
       precision = precision + 1;
    end
    compute_nr_basis_functions = @(knot_vector, p) size(knot_vector, 2) - p - 1;
    mesh = @(a, c) [a:(c-a)/precision:c];
    
    p = compute_p(knot_vector);
    t = check_sanity(knot_vector, p);
    
    if (~t)
      disp("Poorly constructed knot_vector")
      return
    end
    nr = compute_nr_basis_functions(knot_vector, p);
    
    [n, m] = size(points);
    
    % oczekujemy teraz 3 kolumn (x,y,z)
    if (n ~= nr || m ~= 3)
        disp("Poorly constructed points array - expected N x 3");
        return
    end
    
    x_begin = knot_vector(1);
    x_end = knot_vector(size(knot_vector, 2));
    x = mesh(x_begin, x_end);
    
    res_x = zeros(1, numel(x));
    res_y = zeros(1, numel(x));
    res_z = zeros(1, numel(x)); % nowa składowa Z
    
    for ind = 1:nr
        spline = compute_spline(knot_vector, p, ind, x);
        fprintf('\n');
        res_x = res_x + (spline .* points(ind, 1));
        res_y = res_y + (spline .* points(ind, 2));
        res_z = res_z + (spline .* points(ind, 3)); % Z
    end
    
    plot3(res_x, res_y, res_z, 'r', 'LineWidth', 1.5)
    hold on

    scatter3(points(:, 1), points(:, 2), points(:, 3), 36, 'b', 'filled')
    
    axis equal
    grid on
    view(3)
    hold off;

    function p = compute_p(knot_vector)
      initial = knot_vector(1);
      kvsize = size(knot_vector, 2);
      p = 0;
    
      while (p + 2 <= kvsize) && (initial == knot_vector(p + 2))
        p = p + 1;
      end
      return
    end
    
    function y = compute_splines(knot_vector, p, nr, x)
      y = compute_spline(knot_vector, p, nr, x);
      return
    end
    
    function y = compute_spline(knot_vector, p, nr, x)
      fC = @(x, a, b) (b ~= a) .* (x - a) ./ (b - a);
      fD = @(x, c, d) (d ~= c) .* (d - x) ./ (d - c);
     
      a = knot_vector(nr);
      b = knot_vector(nr + p);
      c = knot_vector(nr + 1);
      d = knot_vector(nr + p + 1);
    
      if (p == 0)
        y = 0 .* (x < a) + 1 .* (a <= x & x <= d) + 0 .* (x > d);
        return
      end
    
      lp = compute_spline(knot_vector, p - 1, nr, x);
      rp = compute_spline(knot_vector, p - 1, nr + 1, x);
    
      if (a == b)
        y1 = 0 .* (x < a) + 1 .* (a <= x & x <= b) + 0 .* (x > b);
      else
        y1 = fC(x, a, b) .* (a <= x & x <= b);
      end
      
      if (c == d)
        y2 = 0 .* (x < c) + 1 .* (c <= x & x <= d) + 0 .* (x > d);
      else
        y2 = fD(x, c, d) .* (c <= x & x <= d);
      end
    
      y = lp .* y1 + rp .* y2;
    
      if nr == 1 && x(1) == knot_vector(1)
          y(1) = 1;
      elseif nr == size(knot_vector, 2) - p - 1 && x(end) == knot_vector(end)  % Last basis function
          y(end) = 1;
      end
      return
    end
    
    function t = check_sanity(knot_vector, p)
      initial = knot_vector(1);
      kvsize = size(knot_vector, 2);
      t = true;
      counter = 1;
    
      for i = 1:p + 1
        if (initial ~= knot_vector(i))
          t = false;
          return
        end
      end
    
      for i = p + 2:kvsize - p - 1
        if (initial == knot_vector(i))
          counter = counter + 1;
          if (counter > p)
            t = false;
            return
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
          return
        end
      end
      
      for i = 1:kvsize - 1
        if (knot_vector(i) > knot_vector(i + 1))
          t = false;
          return
        end
      end
      return
    end
end
