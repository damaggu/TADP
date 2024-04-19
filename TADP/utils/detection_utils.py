import torch

CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

REDUCED_CLASS_NAMES = ['background', 'bicycle', 'bird', 'car', 'cat', 'dog', 'person']

DETECTRON_VOC_CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

voc_classes = ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
               'horse', 'motorcycle', 'person', 'potted plant', 'sheep',
               'sofa', 'train', 'television']


def empty_collate(batch):
    return batch


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def annotations_to_boxes(annotations):
    boxes = []
    labels = []
    scores = [1.0]  # Assuming constant score for training
    for annot in annotations:
        boxes_for_image = []
        labels_for_image = []
        scores_for_image = []
        for obj in annot['annotation']['object']:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            label = obj['name']

            boxes_for_image.append([xmin, ymin, xmax, ymax])
            label = CLASS_NAMES.index(label)
            labels_for_image.append(label)
            scores_for_image.append(1.0)

        boxes.append(torch.tensor(boxes_for_image))
        labels.append(torch.tensor(labels_for_image))
        scores.append(torch.tensor(scores_for_image))

    return boxes, labels, scores


def prep_include_classes(args):
    # use default if no val classes are specified
    if args.include_classes_train is None:
        args.include_classes_train = args.include_classes
    if args.include_classes_val is None:
        args.include_classes_val = args.include_classes

    include_classes_train = args.include_classes_train
    # allow user to specify classes with space with an underscore instead
    include_classes_train = [c.replace('_', ' ') for c in include_classes_train]
    if 'all' in include_classes_train:
        include_classes_train = 'all'

    include_classes_val = args.include_classes_val
    # allow user to specify classes with space with an underscore instead
    include_classes_val = [c.replace('_', ' ') for c in include_classes_val]
    if 'all' in include_classes_val:
        include_classes_val = 'all'

    return include_classes_train, include_classes_val


def create_bounding_boxes_from_masks(segmentation_masks):
    """
    Create bounding boxes for different classes from segmentation masks.

    Parameters:
        segmentation_masks (tensor): Segmentation masks of shape (N, H, W) where N is the number of masks.
        num_classes (int): Total number of classes in the segmentation map.

    Returns:
        torch.Tensor: Tensor containing bounding boxes for different classes.
    """
    bounding_boxes = []

    for mask in segmentation_masks:
        classes = torch.unique(mask)[1:]  # skip background
        bboxes_per_mask = []
        for class_val in classes:
            mask_for_object = mask == class_val
            mask_for_object = mask_for_object.type(torch.uint8)

            # Find coordinates of non-zero elements in the mask
            nonzero_indices = torch.nonzero(mask_for_object, as_tuple=False)

            if nonzero_indices.size(0) == 0:
                # Skip if no pixels of the current class are present in the mask
                continue

            # Calculate bounding box coordinates
            ymin, _ = torch.min(nonzero_indices, dim=0)
            ymax, _ = torch.max(nonzero_indices, dim=0)
            ymin = [ymin[1], ymin[0]]
            ymax = [ymax[1], ymax[0]]

            # coords = [ymin[1], ymin[0], ymax[1], ymax[0]]
            coords = [ymin[0], ymin[1], ymax[0], ymax[1]]
            bbox = torch.tensor(coords)
            bboxes_per_mask.append([class_val, bbox])

        bounding_boxes.append(bboxes_per_mask)

    return bounding_boxes
