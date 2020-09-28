%sd = SIMSTxtData();
bw0 = uint8(squeeze(pred(1,:,:)));

bw1 = imfill(bw0,'holes');
bw2 = imopen(bw1,strel('disk',4));

cc = bwconncomp(bw2,4);
cc_mat = labelmatrix(cc);
cc_rgb = label2rgb(cc_mat,'spring','c','shuffle');

edge = bwperim(bw2);
overlap = imoverlay(sd.rawMat/max(sd.rawMat(:)),edge,[1,.3,.3]);

hf = figure;
imagesc(subplot(231),sd.rawMat); set(gca,'looseInset',[0 0 0 0]); xticks([]); yticks([]);
imagesc(subplot(232),bw0); set(gca,'looseInset',[0 0 0 0]); xticks([]); yticks([]);
imagesc(subplot(233),bw1); set(gca,'looseInset',[0 0 0 0]); xticks([]); yticks([]);
imagesc(subplot(234),bw2); set(gca,'looseInset',[0 0 0 0]); xticks([]); yticks([]);
imagesc(subplot(235),cc_rgb); set(gca,'looseInset',[0 0 0 0]); xticks([]); yticks([]);
imagesc(subplot(236),overlap); set(gca,'looseInset',[0 0 0 0]); xticks([]); yticks([]);

linkaxes(hf.Children,'xy');