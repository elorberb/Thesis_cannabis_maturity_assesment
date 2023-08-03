def calculate_precision_recall_per_segment(gt_dict, pred_dict, iou_threshold):
    TPs = 0  # True positives
    FPs = 0  # False positives
    FNs = 0  # False negatives
    iou_values = []  # list to store IOU values

    gt_segments = np.unique(gt_dict['instance_bitmap'])  # List of unique segments in ground truth
    gt_segments = gt_segments[gt_segments != 0]  # Exclude the background (0)

    matched_gt_segments = set()  # Track which gt segments have been matched

    # Iterate over each predicted segment
    for segment in pred_dict['mask']:
        pred_mask = segment['segmentation']
        max_iou = 0
        best_match_gt_seg = None

        for seg in gt_segments:
            if seg not in matched_gt_segments:  # Only consider unmatched gt segments
                gt_mask = (gt_dict['instance_bitmap'] == seg)  # mask for current gt segment
                intersection = np.logical_and(gt_mask, pred_mask)
                union = np.logical_or(gt_mask, pred_mask)
                iou = np.sum(intersection) / np.sum(union)

                if iou > max_iou:
                    max_iou = iou
                    best_match_gt_seg = seg

        if max_iou >= iou_threshold and best_match_gt_seg is not None:
            TPs += 1  # We have a true positive
            matched_gt_segments.add(best_match_gt_seg)  # This gt segment has been matched
        else:
            FPs += 1  # We have a false positive

        iou_values.append(max_iou)  # Add the max IOU to the list

    # Calculate false negatives by subtracting the number of matched gt segments from total gt segments
    FNs = len(gt_segments) - len(matched_gt_segments)

    precision = TPs / (TPs + FPs) if TPs + FPs > 0 else 0
    recall = TPs / (TPs + FNs) if TPs + FNs > 0 else 0

    return precision, recall, (TPs, FPs, FNs), iou_values


def plot_masks(y_true, y_pred):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    image = y_true['image']
    y_true_mask = y_true['instance_bitmap']
    y_pred_mask = y_pred['mask']

    # Plot Original image
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Plot ground truth
    ax[1].imshow(image)
    ax[1].imshow(y_true_mask, alpha=0.5, cmap='jet')  # Overlays the mask on the image, adjust alpha to your needs
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')

    # Plot prediction
    ax[2].imshow(image)
    show_anns(y_pred_mask)
    ax[2].set_title('Prediction')
    ax[2].axis('off')

    plt.show()


import seaborn as sns


def plot_confusion_matrix(TPs, FPs, FNs):
    confusion_matrix = [[TPs, FPs],
                        [FNs, 0]]  # Note that we don't have TNs in this case

    # Create a heatmap
    plt.figure(figsize=(6, 4))
    heatmap = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')

    # Set labels
    heatmap.xaxis.set_ticklabels(['Positive', 'Negative'])
    heatmap.yaxis.set_ticklabels(['Positive', 'Negative'])
    heatmap.xaxis.tick_top()  # x axis on top
    heatmap.xaxis.set_label_position('top')

    # Add axis labels
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.show()


def plot_iou_histogram(iou_values, bins=10):
    plt.figure(figsize=(10, 6))
    plt.hist(iou_values, bins=bins, edgecolor='black')
    plt.xlabel('IOU')
    plt.ylabel('Frequency')
    plt.title('Histogram of IOU Values')
    plt.show()


def calculate_average_precision_recall(ground_truth, SAM_pred, iou_threshold):
    total_precision = 0
    total_recall = 0
    total_TPs = 0
    total_FPs = 0
    total_FNs = 0
    all_iou_values = []
    num_images = len(ground_truth)

    for name in ground_truth:
        precision, recall, (TPs, FPs, FNs), iou_values = calculate_precision_recall_per_segment(ground_truth[name],
                                                                                                SAM_pred[name],
                                                                                                iou_threshold)
        total_precision += precision
        total_recall += recall
        total_TPs += TPs
        total_FPs += FPs
        total_FNs += FNs
        all_iou_values.extend(iou_values)

    average_precision = total_precision / num_images
    average_recall = total_recall / num_images

    return average_precision, average_recall, (total_TPs, total_FPs, total_FNs), all_iou_values


def plot_avg_precision_recall_curve(gt_dict, pred_dict):
    # Define a range of IOU thresholds
    iou_thresholds = np.linspace(0.3, 1.0, 10)  # Vary IOU from 0.3 to 1.0

    # Initialize lists to store precision and recall values
    precision_values = []
    recall_values = []

    # Iterate through the thresholds and calculate precision and recall
    for iou_thresh in iou_thresholds:
        precision, recall = calculate_average_precision_recall(gt_dict, pred_dict, iou_thresh)
        precision_values.append(precision)
        recall_values.append(recall)

    # Plot the precision-recall curve
    plt.figure(figsize=(10, 7))
    for i, value in enumerate(iou_thresholds):
        plt.plot(recall_values[i], precision_values[i], marker='.')
        plt.text(recall_values[i], precision_values[i], f'{value:.1f}')  # Annotate the IoU threshold value
    plt.title('Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()


def plot_precision_recall_curve(gt_dict, pred_dict):
    # Define a range of IOU thresholds
    iou_thresholds = np.linspace(0.1, 1.0, 50)  # Vary IOU from 0.1 to 1.0

    # Initialize lists to store precision and recall values
    precision_values = []
    recall_values = []

    # Iterate through the thresholds and calculate precision and recall
    for iou_thresh in iou_thresholds:
        precision, recall = calculate_precision_recall_per_segment(gt_dict, pred_dict, iou_thresh)
        precision_values.append(precision)
        recall_values.append(recall)

    # Plot the precision-recall curve
    plt.figure(figsize=(10, 7))
    for i, value in enumerate(iou_thresholds):
        plt.plot(recall_values[i], precision_values[i], marker='.')
        plt.text(recall_values[i], precision_values[i], f'{value:.1f}')  # Annotate the IoU threshold value
    plt.title('Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()
