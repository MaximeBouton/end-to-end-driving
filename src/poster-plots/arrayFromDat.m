function array = arrayFromDat(fname)
fid = fopen(fname);
L = str2double(fid.fgetl());
array = zeros(L,1);
for l = 1:L
    array(l) = str2double(fid.fgetl());
end