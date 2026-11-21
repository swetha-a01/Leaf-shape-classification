function feat = extract_features(rgb, mask, boundary)
  % Rotation-tolerant leaf features for GNU Octave + image package.
  pkg load image;
  mask = logical(mask);
  if nnz(mask) < 20
    error("Leaf mask is empty or too small; inspect segmentation.");
  endif

  [rr, cc] = find(mask);
  area = nnz(mask);
  rmin = min(rr); rmax = max(rr); cmin = min(cc); cmax = max(cc);
  bh = max(1, rmax-rmin+1); bw = max(1, cmax-cmin+1);

  % Boundary perimeter and convex-hull solidity
  perimeter = 1;
  if rows(boundary) > 1
    d = diff([boundary; boundary(1,:)]);
    perimeter = max(1, sum(sqrt(sum(d.^2,2))));
  endif
  circularity = 4*pi*area/(perimeter^2);
  compactness = perimeter^2/(4*pi*area);
  bboxFill = area/(bh*bw);
  bboxRatio = min(bw,bh)/max(bw,bh); % invariant to 90-degree rotation
  try
    hull = convhull(cc, rr);
    hullArea = polyarea(cc(hull), rr(hull));
    solidity = area/max(1,hullArea);
  catch
    solidity = 0;
  end_try_catch

  % Principal-axis covariance eigenvalues: rotation-invariant elongation
  cx = mean(cc); cy = mean(rr);
  centered = [cc-cx, rr-cy];
  cov2 = (centered' * centered) / max(1, rows(centered));
  ev = sort(real(eig(cov2)), 'descend');
  ev(ev < 0) = 0;
  if numel(ev) < 2 || ev(1) < 1e-9
    elongation = 0; axisRatio = 0;
  else
    axisRatio = sqrt(ev(2)/ev(1));
    elongation = 1-axisRatio;
  endif

  % Rotation-invariant distribution of distances from centroid.
  % Quantiles retain shape information without depending on leaf angle.
  rad = sqrt((cc-cx).^2 + (rr-cy).^2);
  rad = sort(rad / max(rad));
  nq = [0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95];
  radialQ = zeros(1,numel(nq));
  for j = 1:numel(nq)
    ix = max(1, min(numel(rad), round(nq(j)*numel(rad))));
    radialQ(j) = rad(ix);
  endfor

  % Normalized silhouette descriptor (8x8 occupancy grid).
  % Align pixels to the principal axis so the descriptor is less sensitive
  % to leaf angle. Average with its 180-degree reversal to reduce ambiguity
  % in the sign of the principal axis.
  covxx = mean((cc-cx).^2);
  covyy = mean((rr-cy).^2);
  covxy = mean((cc-cx).*(rr-cy));
  theta = 0.5 * atan2(2*covxy, covxx-covyy);
  xrot = (cc-cx)*cos(theta) + (rr-cy)*sin(theta);
  yrot = -(cc-cx)*sin(theta) + (rr-cy)*cos(theta);
  xscale = max(abs(xrot));
  yscale = max(abs(yrot));
  if xscale < 1e-9, xscale = 1; endif
  if yscale < 1e-9, yscale = 1; endif
  xn = xrot / xscale;
  yn = yrot / yscale;
  gridSize = 8;
  silhouette = zeros(gridSize, gridSize);
  for p = 1:numel(xn)
    gx = min(gridSize, max(1, floor((xn(p)+1)*gridSize/2)+1));
    gy = min(gridSize, max(1, floor((yn(p)+1)*gridSize/2)+1));
    silhouette(gy,gx) = silhouette(gy,gx) + 1;
  endfor
  silhouette = silhouette / max(1, max(silhouette(:)));
  silhouette180 = rot90(silhouette, 2);
  silhouette = 0.5 * (silhouette + silhouette180);
  silhouetteFeat = reshape(silhouette, 1, []);

  % Color statistics computed only over detected leaf pixels
  im = im2double(rgb);
  R = im(:,:,1); G = im(:,:,2); B = im(:,:,3);
  pixR = R(mask); pixG = G(mask); pixB = B(mask);
  hsv = rgb2hsv(im);
  H = hsv(:,:,1); S = hsv(:,:,2); V = hsv(:,:,3);
  pixH = H(mask); pixS = S(mask); pixV = V(mask);
  colourStats = [mean(pixR), std(pixR), mean(pixG), std(pixG), ...
    mean(pixB), std(pixB), mean(pixH), std(pixH), ...
    mean(pixS), std(pixS), mean(pixV), std(pixV)];
  greenIndex = 2*mean(pixG)-mean(pixR)-mean(pixB);
  redGreenRatio = mean(pixR)/max(0.001,mean(pixG));
  greenBlueRatio = mean(pixG)/max(0.001,mean(pixB));

  hueHist = zeros(1,8);
  for k = 1:8
    lo = (k-1)/8; hi = k/8;
    hueHist(k) = sum(pixH >= lo & pixH < hi)/numel(pixH);
  endfor

  % Shape features first; no raw image width/height or angle-dependent bbox aspect.
  feat = [bboxFill, bboxRatio, circularity, log1p(compactness), ...
          solidity, elongation, axisRatio, radialQ, silhouetteFeat, colourStats, ...
          greenIndex, redGreenRatio, greenBlueRatio, hueHist];
  feat(~isfinite(feat)) = 0;
endfunction

