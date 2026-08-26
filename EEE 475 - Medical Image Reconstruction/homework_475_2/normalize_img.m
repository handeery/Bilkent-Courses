function ima = normalize_img(ima)
    ima = abs(ima);
    ima = ima / max(abs(ima(:)));
end

