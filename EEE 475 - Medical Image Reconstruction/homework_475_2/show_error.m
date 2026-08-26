function show_error(recon, ref)
    err = abs(recon - ref);
    figure
    imshow(err,[0 0.2])
    title('Error Image')
end