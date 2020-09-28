function [ costs ] = areaCost( sourceArea,tarArea )
    nt = size(tarArea,1);
    AREA_FACTOR = 9;
    CUTOFF_AREA_RATIO = 1.25;
    SIZE_L_VAR = 0.85;
    SIZE_H_VAR = 1.5;
    ratio = tarArea(:)' ./ sourceArea;
    areaFrac = (sourceArea - min(sourceArea))/(max(sourceArea)-min(sourceArea));
    I = areaFrac > 0.5;
    areaFracModify = 2*(1-SIZE_L_VAR)*areaFrac + SIZE_L_VAR;
    areaFracModify(I) = 2*(SIZE_H_VAR-1)*areaFrac(I) + 2 - SIZE_H_VAR;
    costs = max(ratio,1./ratio).^repmat(AREA_FACTOR*areaFracModify,[1,nt])-1;
    costs = 1/(CUTOFF_AREA_RATIO^AREA_FACTOR-1) * costs;
end

