function predict_leaf()
  pkg load image;
  modelPath = fullfile(pwd(),"output","leaf_model.mat");
  if ~exist(modelPath,"file"), error("Model not found. Run train_model first."); endif
  loaded=load(modelPath); model=loaded.model;
  [filename,pathname]=uigetfile({"*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff","Leaf images"},"Select a leaf image");
  if isequal(filename,0), fprintf("No image selected.\n"); return; endif
  imagePath=fullfile(pathname,filename);
  original=imread(imagePath);
  [rgb,mask,boundary]=preprocess_leaf(original);
  feat=extract_features(rgb,mask,boundary);
  if numel(feat)~=model.feature_count
    error("Feature mismatch: model expects %d, prediction generated %d.",model.feature_count,numel(feat));
  endif
  q=(feat-model.feature_mean)./model.feature_std;
  d=sqrt(sum(bsxfun(@minus,model.features,q).^2,2));
  [sortedD,order]=sort(d); k=min(model.K,numel(order));
  nearest=order(1:k); votes=model.labels(nearest);
  counts=zeros(1,numel(model.class_names));
  for j=1:numel(votes)
    ix=find(strcmp(model.class_names,votes{j}),1);
    if ~isempty(ix), counts(ix)=counts(ix)+1; endif
  endfor
  wi=find(counts==max(counts)); predicted=model.class_names{wi(1)};
  fprintf("\nImage: %s\nPrediction: %s\nNeighbour vote: %d/%d (not a calibrated probability)\nNearest distance: %.4f\n", ...
    imagePath,predicted,max(counts),k,sortedD(1));
  fprintf("Nearest examples:\n");
  for j=1:k
    fprintf("  %-14s distance %.4f | %s\n",votes{j},sortedD(j),model.image_paths{order(j)});
  endfor

  figure("Name","Leaf Classification Diagnostics");
  subplot(2,2,1); imshow(original); title("Original");
  subplot(2,2,2); imshow(mask); title("Detected mask");
  subplot(2,2,3); imshow(rgb); hold on;
  if ~isempty(boundary), plot(boundary(:,2),boundary(:,1),"r-","linewidth",1.5); endif
  title("Boundary"); hold off;
  subplot(2,2,4); axis off;
  text(0.02,0.80,["Prediction: " predicted],"fontsize",12,"fontweight","bold");
  text(0.02,0.60,sprintf("Neighbour vote: %d/%d",max(counts),k),"fontsize",11);
  text(0.02,0.40,sprintf("Nearest distance: %.3f",sortedD(1)),"fontsize",11);
  text(0.02,0.20,"Inspect the mask if prediction is wrong.","fontsize",10);
endfunction
