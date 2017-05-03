load twolayertwotwo.mat convnet
file = uigetfile('*.bmp');
if isequal(file,0)
    errordlg('select any input image');
else
    testKnuckleData=imageDatastore(file);
      
    [Ypred,scores]= classify(convnet,testKnuckleData);
     
    YYpred= predict(convnet,testKnuckleData);
    





errorval=Ypred(1,1);
num = int8(90.00);
count = 0;
for da = 1:50
c=int8(scores(1,da)*100);
if c > num
    count = count + 1 ;
end 
end

if count == 1
    helpdlg('authorized');
else
     helpdlg('unauthorized');
end
end
