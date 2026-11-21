function train_model()
  pkg load image;
  root = pwd();
  dataDir = fullfile(root, "dataset");
  outDir = fullfile(root, "output");
  if ~exist(outDir, "dir"), mkdir(outDir); endif
  classNames = {"Mango", "Neem", "Betel_Leaf", "Custard_Apple", "Hibiscus", "Lemon"};
  extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"};
  X = []; labels = {}; paths = {};
  fprintf("\n--- Loading dataset and extracting rotation-tolerant features ---\n");

  for c = 1:numel(classNames)
    classDir = fullfile(dataDir, classNames{c});
    if ~exist(classDir, "dir"), error("Missing class folder: %s", classDir); endif
    listing = [];
    for e = 1:numel(extensions)
      listing = [listing; dir(fullfile(classDir, extensions{e}))];
    endfor
    fprintf("%s: %d files found\n", classNames{c}, numel(listing));
    for i = 1:numel(listing)
      imagePath = fullfile(classDir, listing(i).name);
      try
        original = imread(imagePath);
        [rgb, mask, boundary] = preprocess_leaf(original);
        f = extract_features(rgb, mask, boundary);
        if any(~isfinite(f)), fprintf("Skipped invalid features: %s\n", imagePath); continue; endif
        X(end+1,:) = f;
        labels{end+1,1} = classNames{c};
        paths{end+1,1} = imagePath;
      catch err
        fprintf("Skipped %s (%s)\n", listing(i).name, err.message);
      end_try_catch
    endfor
  endfor

  if rows(X) < 6, error("Too few usable images. Check folders and segmentation."); endif
  n = rows(X); Kvalues = [1 3 5 7 9];
  Kvalues = Kvalues(Kvalues < n);
  allPred = cell(numel(Kvalues), n);
  allAcc = zeros(1,numel(Kvalues));

  % LOO: standardization is fitted separately inside each fold.
  fprintf("\nLOO comparison (each fold scales using training rows only):\n");
  for ki = 1:numel(Kvalues)
    K = Kvalues(ki); pred = cell(n,1); correct = 0;
    for i = 1:n
      tr = true(n,1); tr(i) = false;
      Xtr = X(tr,:); ytr = labels(tr);
      muF = mean(Xtr,1); sdF = std(Xtr,0,1); sdF(sdF < 1e-8) = 1;
      Xtrz = bsxfun(@rdivide, bsxfun(@minus,Xtr,muF),sdF);
      q = (X(i,:) - muF) ./ sdF;
      d = sqrt(sum(bsxfun(@minus,Xtrz,q).^2,2));
      [~,ord] = sort(d);
      votes = ytr(ord(1:min(K,numel(ord))));
      pred{i} = majority_label(votes,classNames);
      if strcmp(pred{i},labels{i}), correct = correct+1; endif
    endfor
    allPred(ki,:) = pred';
    allAcc(ki) = 100*correct/n;
    fprintf("  K=%d: %.2f%% (%d/%d)\n", K, allAcc(ki), correct, n);
  endfor

  % Pick K by LOO accuracy; deterministic tie-break favours smaller K.
  bestAcc = max(allAcc);
  bestCandidates = find(allAcc == bestAcc);
  [~,smallest] = min(Kvalues(bestCandidates));
  bestIndex = bestCandidates(smallest);
  bestK = Kvalues(bestIndex);
  pred = allPred(bestIndex,:)';

  C = zeros(numel(classNames)); correct = 0;
  for i = 1:n
    a = find(strcmp(classNames,labels{i}),1);
    p = find(strcmp(classNames,pred{i}),1);
    C(a,p) = C(a,p)+1;
    if a == p, correct = correct+1; endif
  endfor
  accuracy = 100*correct/n;
  fprintf("\nSelected K=%d using LOO score: %.2f%% (%d/%d)\n",bestK,accuracy,correct,n);
  fprintf("Confusion matrix: rows=actual, columns=predicted\n");
  fprintf("%18s",""); for c=1:numel(classNames), fprintf("%14s",classNames{c}); endfor; fprintf("\n");
  for r=1:numel(classNames), fprintf("%18s",classNames{r}); fprintf("%14d",C(r,:)); fprintf("\n"); endfor

  mu = mean(X,1); sigma = std(X,0,1); sigma(sigma < 1e-8) = 1;
  Xz = bsxfun(@rdivide,bsxfun(@minus,X,mu),sigma);
  model.class_names = classNames; model.features = Xz; model.raw_features = X;
  model.labels = labels; model.image_paths = paths; model.feature_mean = mu;
  model.feature_std = sigma; model.K = bestK; model.loo_accuracy = accuracy;
  model.confusion_matrix = C; model.feature_count = columns(X);
  model.candidate_K = Kvalues; model.candidate_accuracy = allAcc;
  save(fullfile(outDir,"leaf_model.mat"),"model");

  fid = fopen(fullfile(outDir,"loo_predictions.csv"),"w");
  fprintf(fid,"image_path,actual,predicted,correct\n");
  for i=1:n
    fprintf(fid,'"%s",%s,%s,%d\n',paths{i},labels{i},pred{i},strcmp(labels{i},pred{i}));
  endfor
  fclose(fid);
  fprintf("\nSaved model to output/leaf_model.mat\nSaved validation rows to output/loo_predictions.csv\n");
  fprintf("Note: LOO is internal validation; test on separate leaves/photos too.\n");
endfunction

function winner = majority_label(votes, classNames)
  counts = zeros(1,numel(classNames));
  for j=1:numel(votes)
    idx=find(strcmp(classNames,votes{j}),1);
    if ~isempty(idx), counts(idx)=counts(idx)+1; endif
  endfor
  winners=find(counts==max(counts));
  winner=classNames{winners(1)};
endfunction
