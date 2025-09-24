
##=======================================================================================================
# Requirements:
# - Python 3.x
# - SimpleITK, numpy, tifffile
# Usage:
# 1) Adjust the paths path_T1_original, path_T2_original, and path_T2_absorcion_original to your files.
# 2) Optional: adjust roi_size and offsets to center the ROI.
# 3) Run: python this_script.py
# What it does:
# - Loads T1 (fixed) and T2 (moving) volumes, plus a T2 absorption image.
# - Extracts a centered ROI with offsets.
# - Registers T2 onto T1 (translation + affine) using Elastix.
# - Adjusts the ROI transform to apply it to the full image.
# - Applies the same transform to the absorption image as well.
# - Saves ROIs and full results, and reports parameters (translation, scale, rotation).
##=======================================================================================================
import SimpleITK as sitk
import numpy as np
import tifffile
import math

def main():
    # =========================================================================
    # 1. Initial configuration and parameters
    # ===========================================================================
    # Paths of the original images

    path_T1_original = "imagenes/diatom_T1_indice.tif"
    path_T2_original = "imagenes/diatom_T2_indice.tif"
    # Path of the absorption image
    path_T2_absorcion_original = "imagenes/diatom_T2_absorption.tif"
    
    # ROI parameters
    roi_size = 250
    # Calibration offsets for ROI extraction
    offsets = {
        'x': 0,
        'y': 0,
        'z': 0
    }

    # =========================================================================
    # 2. Load full images and configure metadata
    # ===========================================================================
    print("Loading original images...")
    T1_full_array = tifffile.imread(path_T1_original).astype(np.float32)
    T2_full_array = tifffile.imread(path_T2_original).astype(np.float32)
    # New: Load absorption image
    T1_absorcion_full_array = tifffile.imread(path_T2_absorcion_original).astype(np.float32)
    
    # Create SITK images from arrays
    T1_full_sitk = sitk.GetImageFromArray(T1_full_array)
    T2_full_sitk = sitk.GetImageFromArray(T2_full_array)
    # New: Create SITK image for the absorption image
    T1_absorcion_full_sitk = sitk.GetImageFromArray(T1_absorcion_full_array)
    
    # Configure metadata with real voxel spacing (0.1212629, 0.1212629, 0.1212629)
    original_spacing = (0.1212629, 0.1212629, 0.1212629)
    original_origin = (0.0, 0.0, 0.0)
    original_direction = (1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0)
    
    for img in [T1_full_sitk, T2_full_sitk]:
        img.SetSpacing(original_spacing)
        img.SetOrigin(original_origin)
        img.SetDirection(original_direction)
    # New: Configure metadata for absorption image
    T1_absorcion_full_sitk.SetSpacing(original_spacing)
    T1_absorcion_full_sitk.SetOrigin(original_origin)
    T1_absorcion_full_sitk.SetDirection(original_direction)

    # =========================================================================
    # 3. Extract ROI with correct spatial metadata
    # ===========================================================================
    # The function returns the ROI image and the offset (roi_min) used
    def extract_roi(full_image, offsets):
        full_size = full_image.GetSize()
        
        # Compute center and ROI limits
        center = [dim // 2 for dim in full_size]
        roi_min = [
            center[0] - roi_size // 2 + offsets['x'],
            center[1] - roi_size // 2 + offsets['y'],
            center[2] - roi_size // 2 + offsets['z']
        ]
        
        # Ensure ROI limits remain within image boundaries
        roi_min = [max(0, m) for m in roi_min]
        roi_max = [min(full_size[i], roi_min[i] + roi_size) for i in range(3)]
        
        # Extract subvolume and set spatial metadata
        roi_array = sitk.GetArrayFromImage(full_image)[
            roi_min[2]:roi_max[2],
            roi_min[1]:roi_max[1],
            roi_min[0]:roi_max[0]
        ]
        
        roi_img = sitk.GetImageFromArray(roi_array)
        
        # Compute new physical origin based on real voxel spacing
        new_origin = [
            full_image.GetOrigin()[0] + roi_min[0] * full_image.GetSpacing()[0],
            full_image.GetOrigin()[1] + roi_min[1] * full_image.GetSpacing()[1],
            full_image.GetOrigin()[2] + roi_min[2] * full_image.GetSpacing()[2]
        ]
        
        # Apply spatial metadata to ROI
        roi_img.SetSpacing(full_image.GetSpacing())
        roi_img.SetOrigin(new_origin)
        roi_img.SetDirection(full_image.GetDirection())
        
        return roi_img, roi_min

    print("Extracting ROIs...")
    T1_roi, roi_min_fixed = extract_roi(T1_full_sitk, offsets)
    T2_roi, roi_min_moving = extract_roi(T2_full_sitk, offsets)

    # =========================================================================
    # 4. Perform registration on the ROIs
    # ===========================================================================
    print("Performing registration...")
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(T1_roi)
    elastix.SetMovingImage(T2_roi)
    
    # Configure registration parameters
    params = sitk.VectorOfParameterMap()
    
    # Translation parameters
    trans_map = sitk.GetDefaultParameterMap("translation")
    trans_map["AutomaticTransformInitialization"] = ["true"]
    trans_map["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    trans_map["Metric"] = ["AdvancedMattesMutualInformation"]
    trans_map["NumberOfHistogramBins"] = ["50"]
    trans_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    trans_map["LearningRate"] = ["0.25"]
    trans_map["NumberOfIterations"] = ["300"]
    trans_map["ConvergenceThreshold"] = ["1e-6"]
    trans_map["ConvergenceWindowSize"] = ["10"]
    trans_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    trans_map["FinalBSplineInterpolationOrder"] = ["1"]
    params.append(trans_map)
    
    # Affine parameters
    affine_map = sitk.GetDefaultParameterMap("affine")
    affine_map["AutomaticTransformInitialization"] = ["true"]
    affine_map["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    affine_map["Metric"] = ["AdvancedMattesMutualInformation"]
    affine_map["NumberOfHistogramBins"] = ["50"]
    affine_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    affine_map["LearningRate"] = ["0.25"]
    affine_map["NumberOfIterations"] = ["300"]
    affine_map["ConvergenceThreshold"] = ["1e-6"]
    affine_map["ConvergenceWindowSize"] = ["10"]
    affine_map["OptimizerScales"] = ["PhysicalShift"]
    affine_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    affine_map["FinalBSplineInterpolationOrder"] = ["1"]
    affine_map["NumberOfResolutions"] = ["4"]
    params.append(affine_map)
    
    elastix.SetParameterMap(params)
    elastix.Execute()
    
    # =========================================================================
    # 5. Adjust transformation for the full image and apply registration
    # ===========================================================================
    print("Applying transformation to full image...")
    # Get the transform parameter map(s) estimated from ROI registration
    transform_param_maps = elastix.GetTransformParameterMap()
    
    for param_map in transform_param_maps:
        # Adjust transformation parameters to match full image coordinates
        if "TransformParameters" in param_map:
            params_list = [float(x) for x in param_map["TransformParameters"]]
            # Case: translation transform (3 parameters)
            if len(params_list) == 3:
                new_params = params_list
            # Case: affine transform (12 parameters: 9 for matrix, 3 for translation)
            elif len(params_list) == 12:
                M = np.array(params_list[:9]).reshape((3, 3))
                t = np.array(params_list[9:])
                spacing = np.array(T2_full_sitk.GetSpacing())
                roi_offset_phys = np.array(roi_min_moving) * spacing
                t_new = t + (np.eye(3) - M).dot(roi_offset_phys)
                new_params = list(M.flatten()) + list(t_new)
            else:
                new_params = params_list
            param_map["TransformParameters"] = [str(x) for x in new_params]
        
        # Adjust rotation center if defined (subtract ROI physical offset)
        if "CenterOfRotationPoint" in param_map:
            center = [float(x) for x in param_map["CenterOfRotationPoint"]]
            spacing = np.array(T2_full_sitk.GetSpacing())
            roi_offset_phys = np.array(roi_min_moving) * spacing
            new_center = np.array(center) - roi_offset_phys
            param_map["CenterOfRotationPoint"] = [str(x) for x in new_center]
        
        # Update spatial parameters so output corresponds to full fixed image
        fixed_size = T1_full_sitk.GetSize()
        fixed_spacing = T1_full_sitk.GetSpacing()
        fixed_origin = T1_full_sitk.GetOrigin()
        fixed_direction = T1_full_sitk.GetDirection()
        param_map["Size"] = [str(dim) for dim in fixed_size]
        param_map["Spacing"] = [str(spc) for spc in fixed_spacing]
        param_map["Origin"] = [str(org) for org in fixed_origin]
        param_map["Direction"] = [str(d) for d in fixed_direction]
        param_map["DefaultPixelValue"] = ["0.0"]
    
    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(T2_full_sitk)
    transformix.SetTransformParameterMap(transform_param_maps)
    transformix.Execute()
    
    # =========================================================================
    # 6. Save results
    # ===========================================================================
    print("Saving results...")
    
    tifffile.imwrite("images/Roi_diatom_T1_indice.tif", sitk.GetArrayFromImage(T1_roi)) # Cropped T1
    tifffile.imwrite("images/Roi_diatom_T2_indice.tif", sitk.GetArrayFromImage(T2_roi)) # Cropped T2
    tifffile.imwrite("images/Roi_diatom_T2_indice_moved.tif", sitk.GetArrayFromImage(elastix.GetResultImage())) # Registered T2 (ROI)
    result_full = transformix.GetResultImage()
    tifffile.imwrite("images/Full_diatom_T2_indice_moved.tif", sitk.GetArrayFromImage(result_full))
    
    # New: Apply the same transformation to absorption image and save result
    transformix_abs = sitk.TransformixImageFilter()
    transformix_abs.SetMovingImage(T1_absorcion_full_sitk)
    transformix_abs.SetTransformParameterMap(transform_param_maps)
    transformix_abs.Execute()
    result_abs = transformix_abs.GetResultImage()
    tifffile.imwrite("images/Full_diatom_T2_absorption_moved.tif", sitk.GetArrayFromImage(result_abs))
    
    # =========================================================================
    # 7. Print transformation parameters and global displacement
    # ===========================================================================
    print("\n=== Transformation Parameters ===")
    
    # Transformation 0 (Translation)
    if len(transform_param_maps) > 0:
        trans0 = transform_param_maps[0]
        params0 = [float(x) for x in trans0["TransformParameters"]]
        print("Transformation 0 (Translation):")
        print("  Translation (x, y, z):", params0)
    
    # Transformation 1 (Affine)
    if len(transform_param_maps) > 1:
        trans1 = transform_param_maps[1]
        params1 = [float(x) for x in trans1["TransformParameters"]]
        # Extract affine matrix (first 9 parameters) and translation (last 3)
        M = np.array(params1[:9]).reshape((3, 3))
        t = np.array(params1[9:])
        # Compute scale factors (norm of each column)
        scale_factors = [np.linalg.norm(M[:, i]) for i in range(3)]
        # Extract rotation matrix by normalizing each column
        R = np.zeros((3, 3))
        for i in range(3):
            if scale_factors[i] != 0:
                R[:, i] = M[:, i] / scale_factors[i]
        # Function to compute Euler angles (radians) from rotation matrix (ZYX convention)
        def rotationMatrixToEulerAngles(R):
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
            return x, y, z
        euler_angles_rad = rotationMatrixToEulerAngles(R)
        euler_angles_deg = [math.degrees(angle) for angle in euler_angles_rad]
        
        print("\nTransformation 1 (Affine):")
        print("  Translation (x, y, z):", t)
        print("  Scale factors (x, y, z):", scale_factors)
        print("  Rotation (degrees) (x, y, z):", euler_angles_deg)
    
    # Global displacement applied (physical offset of moving image ROI)
    spacing = np.array(T2_full_sitk.GetSpacing())
    roi_offset_phys = np.array(roi_min_moving) * spacing
    print("\nGlobal displacement (ROI physical offset):", roi_offset_phys)
    
    print("\nProcess completed successfully")
    print(f"Original dimensions of T2 (Top): {T2_full_array.shape}")
    print(f"Result dimensions: {sitk.GetArrayFromImage(result_full).shape}")

if __name__ == "__main__":
    main()
