function bigConf = spUtilTestBigImagesWithNet(net,bigIm,globalMean)

if nargin < 3
    globalMean = 0;
end

inputSize = net.blobs('data').shape;
subImgMat = single(spExt.im2colstep(double(bigIm), inputSize(1:3), [floor(inputSize(1:2)./4) , 1]));
subImgMat = shiftdim(reshape(subImgMat',[size(subImgMat,2),inputSize(1:3)]),1);
subImgMat = permute(single(subImgMat(:,:,[3,2,1],:)),[2,1,3,4]) - globalMean;

subImgMat = permute(spUtilTestSmallImagesWithNet(net,subImgMat),[2,1,3,4]);

bigImSIze = size(bigIm);
channelSize = size(subImgMat,3);
% subImgMat = imfilter(subImgMat,fspecial('gaussian',5,1));
perImageWeighting = imresize(fspecial('gaussian',inputSize(1:2),(min(inputSize(1:2)) - 1)./4),inputSize(1:2));
subImgMat = subImgMat .* perImageWeighting;
subImgMat = reshape(shiftdim(subImgMat,3),size(subImgMat,4),[]).';

bigConf = single(spExt.col2imstep(double(subImgMat), [bigImSIze(1:2),channelSize],[inputSize(1:2),channelSize], [floor(inputSize(1:2)./4) , 1]));
% meanImg = spExt.col2imstep(ones(size(subImgMat)), [bigImSIze(1:2),channelSize],[inputSize(1:2),channelSize], [floor(inputSize(1:2)./1) , 1]);
meanImg = spExt.col2imstep(reshape(shiftdim(repmat(perImageWeighting,[1,1,3,size(subImgMat,2)]),3),size(subImgMat,2),[]).', [bigImSIze(1:2),channelSize],[inputSize(1:2),channelSize], [floor(inputSize(1:2)./4) , 1]);
bigConf = bigConf./meanImg;
end


% cpu_cols
% % Number of channels of input image
% channel_in = 2;
% 
% % Size of the sliding windows/kernels
% kernelSize = inputSize(1:3);
% 
% % Stride in x and y direction
% stride = [19, 19];
% 
% % Input image height-width
% img_size = [5000 5000];
% 
% % Build input image
% % Or substitute with your own image matrix
% img = double(randn([img_size, channel_in]));
% 
% 
% %% err chk
% % Run a cpu version of col2im to check the result
% tic
% 
% toc
% 
% tic
% cpu_img = spExt.col2imstep(cpu_cols, [img_size,channel_in], [kernel_size,channel_in], [stride,1]);
% meanImg = spExt.col2imstep(ones(size(cpu_cols)), [img_size,channel_in], [kernel_size,channel_in], [stride,1]);
% cpu_img = cpu_img./meanImg;
% toc
