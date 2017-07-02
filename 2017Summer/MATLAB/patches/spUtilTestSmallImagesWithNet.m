function imgPredict = spUtilTestSmallImagesWithNet(net,smallImageMat)
%%
netOgSize = net.blobs('data').shape;
net.forward({ones(netOgSize,'single')});
netOutSize = net.blob_vec(end).shape;
%%
smallSize = size(smallImageMat);
nImages = smallSize(4);
smallSize = smallSize(1:3);

% batchSize = spUtilFindMaxBatchSizeForNetwork(net); startInd = 1; endInd = 1;
batchSize = 100; startInd = 1; endInd = 1;
net.blobs('data').reshape([smallSize,batchSize])
net.reshape;

imgPredict = nan([netOutSize(1:3),nImages],'single');
%% loop through all images in batches
while endInd < nImages
    if (nImages - startInd + 1) < batchSize
        batchSize = nImages - startInd + 1;
        net.blobs('data').reshape([smallSize,batchSize])
        net.reshape;
    end
    endInd = batchSize + startInd - 1;
    
    imgPredict(:,:,:,startInd:endInd) = cell2mat(net.forward({single(smallImageMat(:,:,:,startInd:endInd))}));
    
    startInd = endInd + 1;
end

%% put network back to original shape
net.blob_vec(1).reshape(netOgSize)
net.reshape;

end
