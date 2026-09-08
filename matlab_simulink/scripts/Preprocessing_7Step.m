function processed = Preprocessing_7Step(image_path, output_path)
% Preprocessing_7Step - Applies a 7-step enhancement to retinal images
% Steps: Green channel, CLAHE, Gaussian Blur, High-pass, Gamma, Laplacian, Norm

    try
        img = imread(image_path);
        
        % 1. Green channel extraction
        if size(img, 3) == 3
            img_g = img(:,:,2);
        else
            img_g = img;
        end
        
        % 2. CLAHE
        clahe_img = adapthisteq(img_g, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
        
        % 3. Gaussian blur
        blurred = imgaussfilt(clahe_img, 1);
        
        % 4. High-pass (original - blurred)
        % Ensure double for subtraction
        high_pass = double(clahe_img) - double(blurred);
        
        % Add back to original to sharpen
        sharpened = double(clahe_img) + high_pass;
        % Clip
        sharpened(sharpened > 255) = 255;
        sharpened(sharpened < 0) = 0;
        
        % 5. Gamma correction
        % Normalize to [0,1] first
        norm_sharp = sharpened / 255.0;
        gamma_corrected = norm_sharp .^ (1/1.2);
        
        % 6. Laplacian sharpening
        lap = del2(gamma_corrected);
        alpha = 0.5;
        lap_sharp = gamma_corrected - alpha * lap;
        
        % 7. Normalization to [0,1]
        processed = mat2gray(lap_sharp);
        
        if nargin > 1 && ~isempty(output_path)
            imwrite(processed, output_path);
        end
        
    catch ME
        warning('Error in preprocessing: %s', ME.message);
        processed = [];
    end
end
