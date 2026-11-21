
function [rgb, mask, boundary] = preprocess_leaf(input_image)
  pkg load image;

  % Read image
  if ischar(input_image)
    original = imread(input_image);
  else
    original = input_image;
  endif

  if ndims(original) == 2
    original = repmat(original, [1 1 3]);
  endif
  if size(original,3) > 3
    original = original(:,:,1:3);
  endif
  original = im2uint8(original);

  % Reduce very large images while preserving aspect ratio
  scale = min(1, 512 / max(size(original,1), size(original,2)));
  if scale < 1
    original = imresize(original, scale);
  endif

  im = im2double(original);
  [H,W,~] = size(im);

  % Estimate background colour using the image borders
  border = [reshape(im(1,:,:), [], 3);
            reshape(im(end,:,:), [], 3);
            reshape(im(:,1,:), [], 3);
            reshape(im(:,end,:), [], 3)];
  bg = median(border, 1);

  % Colour distance from estimated background
  diffimg = sqrt(sum(bsxfun(@minus, im, reshape(bg,1,1,3)).^2, 3));

  borderDiff = [diffimg(1,:), diffimg(end,:), ...
                diffimg(:,1)', diffimg(:,end)'];
  med = median(borderDiff);
  madv = median(abs(borderDiff - med));

  % Green-colour cue; supports white and beige backgrounds
  R = im(:,:,1);
  G = im(:,:,2);
  B = im(:,:,3);

  greenCue = (G > 0.12) & ...
             (G > R * 1.08) & ...
             (G > B * 1.03) & ...
             ((G - R) > 0.025);

  % Combine background difference and green colour evidence
  threshold = max(0.10, med + 3 * madv);
  mask = (diffimg > threshold) | greenCue;

  % Remove small noise and connect small gaps
  mask = imopen(mask, strel("disk", 2, 0));
  mask = imclose(mask, strel("disk", 4, 0));
  mask = imfill(mask, "holes");

  % Retain the largest connected component
  cc = bwlabel(mask);
  if max(cc(:)) == 0
    error("No leaf detected. Try a clearer image with a plain background.");
  endif

  counts = accumarray(cc(cc > 0), 1);
  [~, idx] = max(counts);
  mask = (cc == idx);

  % Validate segmentation
  ratio = nnz(mask) / numel(mask);
  if ratio < 0.002 || ratio > 0.75
    error("Segmentation failed: detected region is too small or too large.");
  endif

  % Crop around the leaf to remove excess whitespace
  [rr, cols] = find(mask);
  pad = round(0.08 * max(max(rr)-min(rr)+1, max(cols)-min(cols)+1));

  r1 = max(1, min(rr)-pad);
  r2 = min(H, max(rr)+pad);
  c1 = max(1, min(cols)-pad);
  c2 = min(W, max(cols)+pad);

  croppedRGB = original(r1:r2, c1:c2, :);
  croppedMask = mask(r1:r2, c1:c2);

  % Pad to a square WITHOUT stretching the leaf
  ch = size(croppedRGB,1);
  cw = size(croppedRGB,2);
  side = max(ch,cw);

  bg8 = uint8(min(255,max(0,round(bg*255))));
  canvas = repmat(reshape(bg8,1,1,3), side, side, 1);
  maskCanvas = false(side,side);

  top = floor((side-ch)/2) + 1;
  left = floor((side-cw)/2) + 1;

  canvas(top:top+ch-1, left:left+cw-1, :) = croppedRGB;
  maskCanvas(top:top+ch-1, left:left+cw-1) = croppedMask;

  % Resize image and mask together
  rgb = imresize(canvas, [256 256]);
  mask = imresize(maskCanvas, [256 256], "nearest") > 0;

  % Clean small resize artefacts
  mask = imfill(mask, "holes");

  cc = bwlabel(mask);
  if max(cc(:)) == 0
    error("Leaf mask became empty during resizing.");
  endif

  counts = accumarray(cc(cc > 0), 1);
  [~, idx] = max(counts);
  mask = (cc == idx);

  % Extract boundary
  boundaries = bwboundaries(mask);
  if isempty(boundaries)
    boundary = zeros(0,2);
  else
    lengths = cellfun(@(x) rows(x), boundaries);
    [~, idx] = max(lengths);
    boundary = boundaries{idx};
  endif
endfunction

