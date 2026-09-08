function results = IQA_Assessment(image_path)
% IQA_Assessment - Performs Image Quality Assessment on retinal images
% Returns a struct with various quality metrics.
% Fallbacks implemented if specialized toolboxes are missing.

    results = struct('brisque_score', NaN, 'niqe_score', NaN, ...
        'blur_variance', NaN, 'entropy', NaN, 'is_gradable', false, 'feedback', '');

    try
        img = imread(image_path);
        
        % Ensure RGB
        if size(img, 3) == 1
            img_rgb = cat(3, img, img, img);
        else
            img_rgb = img;
        end
        
        img_gray = rgb2gray(img_rgb);
        
        % 1. BRISQUE Score (requires Image Processing Toolbox)
        try
            results.brisque_score = brisque(img_rgb);
        catch
            disp('BRISQUE not available, skipping.');
        end
        
        % 2. NIQE Score
        try
            results.niqe_score = niqe(img_rgb);
        catch
            disp('NIQE not available, skipping.');
        end
        
        % 3. Laplacian Variance for Blur
        % A lower variance indicates a blurrier image
        laplacian = del2(double(img_gray));
        results.blur_variance = var(laplacian(:));
        
        % 4. Shannon Entropy
        % Entropy measures the information content
        results.entropy = entropy(img_gray);
        
        % 5. Gradability Check (Heuristics)
        % These thresholds are examples and should be tuned
        min_blur_var = 10.0;
        min_entropy = 4.0;
        
        feedback = [];
        
        if results.blur_variance < min_blur_var
            feedback = [feedback, 'Image is too blurry. '];
        end
        
        if results.entropy < min_entropy
            feedback = [feedback, 'Image lacks contrast/detail (low entropy). '];
        end
        
        if mean(img_gray(:)) < 20
            feedback = [feedback, 'Image is too dark. '];
        end
        
        if isempty(feedback)
            results.is_gradable = true;
            results.feedback = 'Image quality is acceptable.';
        else
            results.is_gradable = false;
            results.feedback = feedback;
        end
        
    catch ME
        results.feedback = ['Error processing image: ', ME.message];
    end
end
