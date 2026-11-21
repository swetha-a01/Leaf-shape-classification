function compare_features()
  % Compare shape-only vs shape+colour features without changing leaf_model.mat.
  % Run from project root: compare_features
  pkg load image;

  datasetDir = 'dataset';
  testDir = 'tests';
  outputDir = 'output';
  classNames = {'Mango','Neem','Betel_Leaf','Custard_Apple','Hibiscus','Lemon'};
  kValues = [1 3 5 7 9];

  if ~exist(datasetDir, 'dir')
    error('Cannot find "%s". Run this script from your project root.', datasetDir);
  endif
  if ~exist(outputDir, 'dir'), mkdir(outputDir); endif

  fprintf('\n=== FEATURE COMPARISON: SHAPE ONLY vs SHAPE + COLOUR ===\n');
  fprintf('Training dataset: %s\n', fullfile(pwd, datasetDir));
  fprintf('This script does NOT overwrite output/leaf_model.mat.\n\n');

  X = [];
  y = [];
  filePaths = {};

  for c = 1:numel(classNames)
    classDir = fullfile(datasetDir, classNames{c});
    if ~exist(classDir, 'dir')
      error('Missing class folder: %s', classDir);
    endif
    listing = dir(classDir);
    count = 0;
    for j = 1:numel(listing)
      if listing(j).isdir, continue; endif
      [~,~,ext] = fileparts(listing(j).name);
      if ~any(strcmpi(ext, {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})), continue; endif
      path = fullfile(classDir, listing(j).name);
      try
        [rgb, mask, boundary] = preprocess_leaf(path);
        f = extract_features(rgb, mask, boundary);
        if isempty(X)
          X = zeros(0, numel(f));
        elseif numel(f) ~= columns(X)
          error('Feature length differs for image: %s', path);
        endif
        X(end+1,:) = f;
        y(end+1,1) = c;
        filePaths{end+1,1} = path;
        count = count + 1;
      catch err
        fprintf('SKIPPED %s: %s\n', path, err.message);
      end_try_catch
    endfor
    fprintf('%-15s %d usable images\n', classNames{c}, count);
  endfor

  n = rows(X);
  if n < 2, error('Not enough training images were loaded.'); endif
  if columns(X) < 82
    error('Expected silhouette features in extract_features.m; got only %d features.', columns(X));
  endif

  % Current extract_features.m order: 7 basic shape + 10 radial + 64 silhouette,
  % followed by 23 colour features. Shape-only is columns 1:81.
  Xshape = X(:,1:81);
  Xfull = X;
  featureSets = {Xshape, Xfull};
  setNames = {'Shape-only', 'Shape + colour'};
  bestK = zeros(1,2);
  bestAcc = zeros(1,2);
  bestPred = cell(1,2);

  for s = 1:2
    F = featureSets{s};
    fprintf('\n--- %s (%d features) ---\n', setNames{s}, columns(F));
    accs = zeros(size(kValues));
    for ki = 1:numel(kValues)
      k = kValues(ki);
      if k >= n, accs(ki) = NaN; continue; endif
      correct = 0;
      for i = 1:n
        trainIdx = true(n,1); trainIdx(i) = false;
        A = F(trainIdx,:); b = y(trainIdx);
        mu = mean(A,1);
        sig = std(A,0,1);
        sig(~isfinite(sig) | sig < 1e-10) = 1;
        A = bsxfun(@rdivide, bsxfun(@minus, A, mu), sig);
        q = (F(i,:) - mu) ./ sig;
        d = sqrt(sum(bsxfun(@minus, A, q).^2, 2));
        [~, order] = sort(d, 'ascend');
        pred = vote_labels(b(order(1:k)), numel(classNames));
        if pred == y(i), correct = correct + 1; endif
      endfor
      accs(ki) = correct/n;
      fprintf('K=%d: %.2f%% (%d/%d)\n', k, 100*accs(ki), correct, n);
    endfor
    maxAcc = max(accs);
    idx = find(accs == maxAcc, 1, 'first');
    bestK(s) = kValues(idx);
    bestAcc(s) = maxAcc;
    fprintf('Best K for %s: %d (LOO %.2f%%)\n', setNames{s}, bestK(s), 100*bestAcc(s));

    % Fit full training set using the selected K and save predictions for external photos.
    mu = mean(F,1);
    sig = std(F,0,1);
    sig(~isfinite(sig) | sig < 1e-10) = 1;
    Z = bsxfun(@rdivide, bsxfun(@minus, F, mu), sig);
    bestPred{s} = struct('mu',mu,'sig',sig,'Z',Z,'y',y,'k',bestK(s),'Fcols',columns(F));
  endfor

  fprintf('\n=== SUMMARY ===\n');
  fprintf('%-18s %12s %8s\n', 'Feature set', 'LOO accuracy', 'Best K');
  fprintf('%-18s %11.2f%% %8d\n', setNames{1}, 100*bestAcc(1), bestK(1));
  fprintf('%-18s %11.2f%% %8d\n', setNames{2}, 100*bestAcc(2), bestK(2));
  if bestAcc(1) > bestAcc(2)
    fprintf('Internal validation favours shape-only; confirm on independent test photos.\n');
  elseif bestAcc(2) > bestAcc(1)
    fprintf('Internal validation favours shape + colour; confirm on independent test photos.\n');
  else
    fprintf('Internal validation ties; use independent test results and additional photos to decide.\n');
  endif

  % Test folder predictions; filename class labels are inferred only for scoring.
  if exist(testDir, 'dir')
    testFiles = dir(testDir);
    csvPath = fullfile(outputDir, 'feature_comparison_external.csv');
    fid = fopen(csvPath, 'w');
    fprintf(fid, 'filename,expected,shape_only_prediction,shape_plus_colour_prediction\n');
    total = 0; correctShape = 0; correctFull = 0;
    fprintf('\n=== EXTERNAL TEST PHOTOS ===\n');
    fprintf('%-24s %-16s %-18s %-20s\n', 'Filename', 'Expected', 'Shape-only', 'Shape + colour');
    fprintf('%s\n', repmat('-',1,82));
    for j = 1:numel(testFiles)
      if testFiles(j).isdir, continue; endif
      [~,~,ext] = fileparts(testFiles(j).name);
      if ~any(strcmpi(ext, {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})), continue; endif
      path = fullfile(testDir, testFiles(j).name);
      try
        [rgb, mask, boundary] = preprocess_leaf(path);
        f = extract_features(rgb, mask, boundary);
        expected = expected_from_filename(testFiles(j).name, classNames);
        predictions = cell(1,2);
        for s = 1:2
          model = bestPred{s};
          if model.Fcols == 81, q = f(1:81); else q = f; endif
          q = (q - model.mu) ./ model.sig;
          d = sqrt(sum(bsxfun(@minus, model.Z, q).^2, 2));
          [~, order] = sort(d, 'ascend');
          p = vote_labels(model.y(order(1:model.k)), numel(classNames));
          predictions{s} = classNames{p};
        endfor
        if isempty(expected), expectedText = 'Unknown'; else expectedText = classNames{expected}; endif
        fprintf('%-24s %-16s %-18s %-20s\n', testFiles(j).name, expectedText, predictions{1}, predictions{2});
        fprintf(fid, '"%s","%s","%s","%s"\n', testFiles(j).name, expectedText, predictions{1}, predictions{2});
        if ~isempty(expected)
          total = total + 1;
          if strcmp(predictions{1}, classNames{expected}), correctShape = correctShape + 1; endif
          if strcmp(predictions{2}, classNames{expected}), correctFull = correctFull + 1; endif
        endif
      catch err
        fprintf('Could not test %s: %s\n', testFiles(j).name, err.message);
      end_try_catch
    endfor
    fclose(fid);
    if total > 0
      fprintf('\nExternal score (filename-labelled images):\n');
      fprintf('Shape-only:       %d/%d = %.2f%%\n', correctShape, total, 100*correctShape/total);
      fprintf('Shape + colour:   %d/%d = %.2f%%\n', correctFull, total, 100*correctFull/total);
    else
      fprintf('\nNo expected labels inferred from filenames; inspect predictions manually.\n');
    endif
    fprintf('Saved comparison CSV: %s\n', csvPath);
  else
    fprintf('\nNo "%s" folder found; external test comparison skipped.\n', testDir);
  endif

  fprintf('\nDONE. Existing output/leaf_model.mat was not changed.\n');
endfunction

function pred = vote_labels(labels, nClasses)
  counts = accumarray(labels(:), 1, [nClasses,1]);
  maxCount = max(counts);
  tied = find(counts == maxCount);
  % Deterministic tie-break: nearest label among the tied classes.
  pred = tied(1);
  for i = 1:numel(labels)
    if any(tied == labels(i))
      pred = labels(i);
      return;
    endif
  endfor
endfunction

function idx = expected_from_filename(filename, classNames)
  name = lower(filename);
  idx = [];
  % Explicit aliases prevent class-name punctuation from affecting matching.
  aliases = {{'mango'}, {'neem'}, {'betel','betel_leaf'}, ...
             {'custard','custard_apple'}, {'hibiscus'}, {'lemon'}};
  for c = 1:numel(classNames)
    a = aliases{c};
    for j = 1:numel(a)
      if ~isempty(strfind(name, a{j}))
        idx = c;
        return;
      endif
    endfor
  endfor
endfunction


