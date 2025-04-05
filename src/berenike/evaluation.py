import os
import geopandas as gpd

def evaluate_detections(save_path: str, ground_truth_path: str, detections_path: str, validation_area_path: str, buffer_distance: int) -> None:
    """
    Compares geojson files of ground truth and detection points. Calculates precision, recall, and F1 score within the evaluation area.

    Args:
        save_path (str): Path to save the results.
        ground_truth_path (str): Path to the ground truth GeoJSON file.
        detections_path (str): Path to the detections GeoJSON file.
        validation_area_path (str): Path to the validation area GeoJSON file.
        buffer_distance (int): Buffer distance in meters for matching points.
    """
    # Load ground truth and detection points
    ground_truth = gpd.read_file(ground_truth_path)
    detections = gpd.read_file(detections_path)
    validation_area = gpd.read_file(validation_area_path)

    ground_truth_crs = ground_truth.crs
    
    # Check if the CRS of the datasets match
    validation_area = validation_area.to_crs(ground_truth_crs)
    detections = detections.to_crs(ground_truth_crs)

    # Ensure the validation area is a single (multi)polygon
    if len(validation_area) > 1:
        validation_area = validation_area.unary_union
    else:
        validation_area = validation_area.geometry.iloc[0]

    # Filter ground truth and detections to those within the validation area
    ground_truth = ground_truth[ground_truth.geometry.within(validation_area)]
    detections = detections[detections.geometry.within(validation_area)]

    # Estimate an appropriate UTM CRS based on the ground truth data
    utm_crs = ground_truth.estimate_utm_crs()

    # Reproject both datasets to the estimated UTM CRS
    ground_truth = ground_truth.to_crs(utm_crs)
    detections = detections.to_crs(utm_crs)

    # Create a buffer around each ground truth point
    ground_truth['buffer'] = ground_truth.geometry.buffer(buffer_distance)
    detections['buffer'] = detections.geometry.buffer(buffer_distance)

    # Initialize lists to store matched and unmatched detections and annotations
    true_positives = []
    false_positives = []
    matched_annotations = []
    unmatched_annotations = []

    # Iterate over each detection to determine if it matches any ground truth point
    for det_idx, detection in detections.iterrows():
        # Check if the detection is within any ground truth buffer
        is_matched = ground_truth.geometry.intersects(detection['buffer']).any()
        if is_matched:
            true_positives.append(detection)
        else:
            false_positives.append(detection)

    # Determine matched and unmatched ground truth annotations
    for idx, gt in ground_truth.iterrows():
        # Check if any detection is within the ground truth buffer
        is_matched = detections.geometry.intersects(gt['buffer']).any()
        if is_matched:
            matched_annotations.append(gt)
        else:
            unmatched_annotations.append(gt)

    # Convert lists to GeoDataFrames
    true_positives_gdf = gpd.GeoDataFrame(true_positives, crs=utm_crs).to_crs(ground_truth_crs)
    false_positives_gdf = gpd.GeoDataFrame(false_positives, crs=utm_crs).to_crs(ground_truth_crs)
    matched_annotations_gdf = gpd.GeoDataFrame(matched_annotations, crs=utm_crs).to_crs(ground_truth_crs)
    unmatched_annotations_gdf = gpd.GeoDataFrame(unmatched_annotations, crs=utm_crs).to_crs(ground_truth_crs)

    # Compute precision, recall, and F1 score
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(unmatched_annotations)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Display results
    print("Within evaluation area:")
    print(f'True Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'False Negatives: {fn}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

    # empty line for better readability
    print()

    # Save to GeoJSON files
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving true positives, false positives, matched and unmatched annotations to {save_path=}")
    true_positives_gdf.to_file(f"{save_path}/true_positives.geojson", driver='GeoJSON')
    false_positives_gdf.to_file(f"{save_path}/false_positives.geojson", driver='GeoJSON')
    matched_annotations_gdf.to_file(f"{save_path}/matched_annotations.geojson", driver='GeoJSON')
    unmatched_annotations_gdf.to_file(f"{save_path}/unmatched_annotations.geojson", driver='GeoJSON')