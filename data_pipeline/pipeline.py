from .preprocessing import extract_tesseract_information
from .preprocessing import get_chargrid
from .preprocessing import extract_class_bounding_boxes
from .preprocessing import get_reduced_output

from .preprocessing_bis import get_max_reduce
from .preprocessing_bis import get_img_reduced
from .preprocessing_bis import reduce_pd_bbox

from .preprocessing_ter import discard_digits_with_low_occurence
from .preprocessing_ter import convert_to_1h
from .preprocessing_ter import resize_to_target
from .preprocessing_ter import extract_anchor_mask
from .preprocessing_ter import extract_anchor_coordinates

import numpy as np

def get_one_hot_encoded_chargrid(image_file_name: str) -> dict:
    """
    Function to convert a given image into one hot encoded chargrid
    :param image_file_name: str
    :return a dictionary containing image chargrid, anchor coordinates, category classes:
    """

    '''
    Get image chargrid, ground truth of class texts in the image and bounding boxes info.
    '''
    document_text_dataframe, img_shape = extract_tesseract_information(image_file_name)
    chargrid_pd = get_chargrid(document_text_dataframe)
    gt_pd = extract_class_bounding_boxes(image_file_name)
    chargrid_np, gt_np, gt_pd = get_reduced_output(chargrid_pd, gt_pd, img_shape)

    '''
    Reduce image size as much as possible without losing too much information from image chargrids.
    Also adjust bounding boxes coordinates accordingly.
    '''
    if np.shape(chargrid_np) != (0, 0):
        reduce_y, padding_top, padding_bot = get_max_reduce(chargrid_np, 0)
        reduce_x, padding_left, padding_right = get_max_reduce(chargrid_np, 1)

        chargrid_np = get_img_reduced(chargrid_np, reduce_x, reduce_y,
                                      padding_left, padding_right, padding_top, padding_bot)
        gt_np = get_img_reduced(gt_np, reduce_x, reduce_y, padding_left, padding_right, padding_top, padding_bot)
        gt_pd = reduce_pd_bbox(gt_pd, padding_left, padding_top, reduce_x, reduce_y)
    else:
        raise ValueError('Cannot process empty image.')

    '''
    Resize all images to same shape (256 height and 128 width).
    Converts all inputs and ground truths to one-hot encoding.
    '''
    chargrid_np = discard_digits_with_low_occurence([chargrid_np])[0]
    chargrid_np_one_hot, gt_np_one_hot = convert_to_1h(chargrid_np, gt_np)
    chargrid_np_one_hot, gt_np_one_hot = resize_to_target(chargrid_np_one_hot, gt_np_one_hot)
    np_bbox_anchor_mask = extract_anchor_mask(gt_pd, np.shape(chargrid_np))
    np_bbox_anchor_coord = extract_anchor_coordinates(gt_pd, np.shape(chargrid_np))

    return {
        'chargrid': chargrid_np_one_hot,
        'ground_truth': gt_np_one_hot,
        'anchor_coords': np_bbox_anchor_coord,
        'anchor_mask': np_bbox_anchor_mask
    }


if __name__ == '__main__':
    single_file_converted = get_one_hot_encoded_chargrid('000.jpg')
    print(
        single_file_converted['chargrid'].shape,
        single_file_converted['ground_truth'].shape,
        single_file_converted['anchor_coords'].shape,
        single_file_converted['anchor_mask'].shape
    )
