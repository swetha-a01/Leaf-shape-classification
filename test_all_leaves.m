function test_all_leaves(testDir)
  % Test every image in one folder using the saved Octave leaf model.
  % Run from the project root: test_all_leaves('tests')
  pkg load image;

  if nargin < 1 || isempty(testDir)
    testDir = fullfile(pwd(), 'tests');
  elseif ~is_absolute_path(testDir)
    testDir = fullfile(pwd(), testDir);
  endif

  modelPath = fullfile(pwd(), 'output', 'leaf_model.mat');
  if ~exist(modelPath, 'file')
    error('Model not found: %s\nRun train_model first from the project root.', modelPath);
  endif
  if ~exist(testDir, 'dir')
    error('Test folder not found: %s', testDir);
  endif

  loaded = load(modelPath);
  if ~isfield(loaded, 'model')
    error('The MAT file does not contain a variable named model.');
  endif
  model = loaded.model;

  extensions = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'};
  files = [];
  for e = 1:numel(extensions)
    files = [files; dir(fullfile(testDir, extensions{e}))];
  endfor
  if isempty(files)
    error('No JPG/PNG/BMP/TIF images found in %s', testDir);
  endif

  fprintf('\nTesting %d image(s) in: %s\n', numel(files), testDir);
  fprintf('Model K=%d | Internal LOO accuracy=%.2f%%\n\n', model.K, model.loo_accuracy);
  fprintf('%-24s %-18s %-18s %s\n', 'IMAGE', 'EXPECTED', 'PREDICTED', 'RESULT');
  fprintf('%s\n', repmat('-',1,82));

  results = {};
  correctCount = 0;
  labelledCount = 0;
  for i = 1:numel(files)
    filePath = fullfile(testDir, files(i).name);
    expected = expected_from_filename(files(i).name, model.class_names);
    try
      original = imread(filePath);
      [rgb, mask, boundary] = preprocess_leaf(original);
      feat = extract_features(rgb, mask, boundary);
      if numel(feat) ~= columns(model.raw_features)
        error('Feature count mismatch: model has %d, image has %d.', columns(model.raw_features), numel(feat));
      endif

      z = (feat - model.feature_mean) ./ model.feature_std;
      d = sqrt(sum(bsxfun(@minus, model.features, z).^2, 2));
      [sortedD, order] = sort(d, 'ascend');
      k = min(model.K, numel(order));
      votes = model.labels(order(1:k));
      predicted = majority_label_local(votes, model.class_names);

      if isempty(expected)
        result = 'unknown filename';
      elseif strcmp(expected, predicted)
        result = 'CORRECT';
        correctCount = correctCount + 1;
        labelledCount = labelledCount + 1;
      else
        result = 'WRONG';
        labelledCount = labelledCount + 1;
      endif
      fprintf('%-24s %-18s %-18s %s\n', truncate_text(files(i).name,24), ...
              blank_if_empty(expected), predicted, result);

      fprintf('  Nearest training examples:\n');
      for j = 1:min(3,numel(order))
        fprintf('    %d) %-16s distance=%.4f | %s\n', j, ...
          model.labels{order(j)}, sortedD(j), model.image_paths{order(j)});
      endfor
      fprintf('\n');
      results(end+1,:) = {files(i).name, expected, predicted, result, sortedD(1)};
    catch err
      fprintf('%-24s ERROR: %s\n\n', truncate_text(files(i).name,24), err.message);
      results(end+1,:) = {files(i).name, expected, 'ERROR', err.message, NaN};
    end_try_catch
  endfor

  if labelledCount > 0
    fprintf('Filename-based score: %d/%d = %.2f%%\n', correctCount, labelledCount, 100*correctCount/labelledCount);
    fprintf('Note: this score is only valid when filenames clearly identify the true class.\n');
  else
    fprintf('Could not infer expected classes from filenames; inspect predictions manually.\n');
  endif

  outPath = fullfile(pwd(), 'output', 'external_test_results.csv');
  if exist(fullfile(pwd(),'output'),'dir')
    fid = fopen(outPath,'w');
    if fid >= 0
      fprintf(fid,'filename,expected,predicted,result,nearest_distance\n');
      for i=1:rows(results)
        fprintf(fid,'"%s","%s","%s","%s",%.6f\n', results{i,1}, results{i,2}, results{i,3}, results{i,4}, results{i,5});
      endfor
      fclose(fid);
      fprintf('Saved results to %s\n', outPath);
    endif
  endif
endfunction

function expected = expected_from_filename(filename, classNames)
  name = lower(filename);
  expected = '';
  aliases = {{'betel','betel_leaf'}, {'custard','custard_apple'}, ...
             {'hibiscus','hibiscus'}, {'lemon','lemon'}, ...
             {'mango','mango'}, {'neem','neem'}};
  for i=1:numel(aliases)
    if ~isempty(strfind(name, aliases{i}{1}))
      target = aliases{i}{2};
      for j=1:numel(classNames)
        if strcmpi(classNames{j},target)
          expected = classNames{j};
          return;
        endif
      endfor
    endif
  endfor
endfunction

function winner = majority_label_local(votes, classNames)
  counts = zeros(1,numel(classNames));
  for j=1:numel(votes)
    idx=find(strcmp(classNames,votes{j}),1);
    if ~isempty(idx), counts(idx)=counts(idx)+1; endif
  endfor
  winners=find(counts==max(counts));
  winner=classNames{winners(1)};
endfunction

function s = blank_if_empty(s)
  if isempty(s), s='(not found)'; endif
endfunction

function out = truncate_text(in, maxLen)
  if numel(in) <= maxLen
    out = in;
  else
    out = [in(1:maxLen-3) '...'];
  endif
endfunction

function tf = is_absolute_path(p)
  tf = ~isempty(p) && (p(1) == '/' || (numel(p) >= 2 && p(2) == ':'));
endfunction

