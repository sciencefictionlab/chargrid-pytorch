from typing import Union, Dict, Any

from .preprocessing import extract_tesseract_information, get_groundTruth, get_final_groundtruth
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
from .setup import check_dataset_dir_present

import numpy as np
import os
from alive_progress import alive_bar
import multiprocessing
from multiprocessing import Queue
from datetime import datetime



dir_img = os.getenv('DIR_IMG')
dir_boxes = os.getenv('DIR_BOXES')
dir_classes = os.getenv('DIR_CLASSES')

outdir_np_chargrid = os.getenv('DIR_NP_CHARGRID')
outdir_png_chargrid = os.getenv('DIR_PNG_CHARGRID')
outdir_np_gt = os.getenv('DIR_NP_GT')
outdir_png_gt = os.getenv('DIR_PNG_GT')
outdir_pd_bbox = os.getenv('DIR_PD_BBOX')

outdir_np_chargrid_reduced = os.getenv('DIR_NP_CHARGRID_REDUCED')
outdir_png_chargrid_reduced = os.getenv('DIR_PNG_CHARGRID_REDUCED')
outdir_np_gt_reduced = os.getenv('DIR_NP_GT_REDUCED')
outdir_png_gt_reduced = os.getenv('DIR_PNG_GT_REDUCED')
outdir_pd_bbox_reduced = os.getenv('DIR_PD_BBOX_REDUCED')


outdir_np_chargrid_1h = os.getenv('DIR_NP_CHARGRID_1H')
outdir_np_gt_1h = os.getenv('DIR_NP_GT_1H')
outdir_np_bbox_anchor_mask = os.getenv('DIR_NP_BBOX_ANCHOR_MASK')
outdir_np_bbox_anchor_coord = os.getenv('DIR_NP_BBOX_ANCHOR_COORD')



def get_one_hot_encoded_chargrid(image_file_name: str) -> Union[int, Dict[str, Any]]:
    print('----> ' + image_file_name + '*****')
    """
    Function to convert a given image into one hot encoded chargrid
    :param image_file_name: str
    :return a dictionary containing image chargrid, anchor coordinates, category classes:
    """

    '''
    Get image chargrid, ground truth of class texts in the image and bounding boxes info.
    '''
    # document_text_dataframe, img_shape = extract_tesseract_information(image_file_name)
    # chargrid_pd = get_chargrid(document_text_dataframe)
    # gt_pd = extract_class_bounding_boxes(image_file_name)
    # chargrid_np, gt_np, gt_pd = get_reduced_output(chargrid_pd, gt_pd, img_shape)

    document_text_dataframe, img_shape = extract_tesseract_information(image_file_name)
    chargrid_np = get_chargrid(document_text_dataframe, img_shape)
    gt_pd = get_groundTruth(image_file_name)
    gt_pd, gt_np, chargrid_np = get_final_groundtruth(gt_pd, chargrid_np, img_shape)


    #save first output
    np.save(os.path.join(outdir_np_chargrid, image_file_name).replace("jpg", "npy"), chargrid_np)
    np.save(os.path.join(outdir_np_gt, image_file_name).replace("jpg", "npy"), gt_np)
    gt_pd.to_pickle(os.path.join(outdir_pd_bbox, image_file_name).replace("jpg", "pkl"))



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


        # save second output

        np.save(os.path.join(outdir_np_chargrid_reduced, image_file_name), chargrid_np)
        np.save(os.path.join(outdir_np_gt_reduced, image_file_name), gt_np)
        gt_pd.to_pickle(os.path.join(outdir_pd_bbox_reduced, image_file_name).replace("npy", "pkl"))

    else:
        print('Cannot process empty image --> .'.format(image_file_name))
        return -1

    '''
    Resize all images to same shape (256 height and 128 width).
    Converts all inputs and ground truths to one-hot encoding.
    '''
    chargrid_np = discard_digits_with_low_occurence([chargrid_np])[0]
    chargrid_np_one_hot, gt_np_one_hot = convert_to_1h(chargrid_np, gt_np)
    chargrid_np_one_hot, gt_np_one_hot = resize_to_target(chargrid_np_one_hot, gt_np_one_hot)
    np_bbox_anchor_mask = extract_anchor_mask(gt_pd, np.shape(chargrid_np))
    np_bbox_anchor_coord = extract_anchor_coordinates(gt_pd, np.shape(chargrid_np))

    # save last output
    np.save(os.path.join(outdir_np_chargrid_1h, image_file_name), chargrid_np_one_hot)
    np.save(os.path.join(outdir_np_gt_1h, image_file_name), gt_np_one_hot)
    np.save(os.path.join(outdir_np_bbox_anchor_coord, image_file_name), np_bbox_anchor_coord)
    np.save(os.path.join(outdir_np_bbox_anchor_mask, image_file_name), np_bbox_anchor_mask)


    return {
        'chargrid': chargrid_np_one_hot,
        'ground_truth': gt_np_one_hot,
        'anchor_coords': np_bbox_anchor_coord,
        'anchor_mask': np_bbox_anchor_mask
    }

def get_one_hot_encoded_chargrid_for_list(image_list: list, out_queue: Queue, show_progress: bool = False):
    processed = []
    if show_progress:
        with alive_bar(len(image_list), spinner='notes2') as bar:
            for file in image_list:
                processed.append(get_one_hot_encoded_chargrid(file))
                bar()
    else:
        for file in image_list:
            processed.append(get_one_hot_encoded_chargrid(file))

    processed = [x for x in processed if x != -1]
    out_queue.put(processed)


def process_dataset(dataset_dir_path: str, num_workers: int = 0, save_np_file:bool = False) -> list:
    if num_workers == 0:
        num_workers = os.cpu_count()//2

    list_filenames = [f for f in os.listdir(dataset_dir_path) if os.path.isfile(os.path.join(dataset_dir_path, f))]
    #list_filenames = list_filenames[:10]
    dataset_length = len(list_filenames)

    if dataset_length < 1:
        raise ValueError('dataset directory is empty')

    processes = []
    processed_dataset = Queue()
    step_length = dataset_length//num_workers 
    start_index = 0

    for i in range(num_workers):
        # batch of images handled by each worker
        end_index = (i+1) * step_length

        tmp_list = list_filenames[start_index:] if end_index > dataset_length else list_filenames[start_index:end_index]

        if len(tmp_list) > 0:
            process = multiprocessing.Process(target=get_one_hot_encoded_chargrid_for_list, args=(tmp_list, processed_dataset,))
            processes.append(process)

        start_index = start_index + step_length

    time_then = datetime.now()
    for process in processes:
        process.start()
        print('spawned process with pid: {} '.format(process.pid))

    result = []
    for i in range(len(processes)):
        result.append(processed_dataset.get())

    for process in processes:
        process.join()
        print('process: {} finished'.format(process.pid))
        #print('processed dataset length: {}'.format(len(processed_dataset)))

    print("total time taken for file parsing: {}".format((datetime.now() - time_then).total_seconds()))
    return result


if __name__ == '__main__':
    # single_file_converted = get_one_hot_encoded_chargrid('000.jpg')
    # print(
    #     single_file_converted['chargrid'].shape,
    #     single_file_converted['ground_truth'].shape,
    #     single_file_converted['anchor_coords'].shape,
    #     single_file_converted['anchor_mask'].shape
    # )

    check_dataset_dir_present()

    dataset_list = process_dataset(os.getenv('DIR_IMG'), 6)
    print(len(dataset_list))
    print(len(dataset_list[0]))
    print(len(dataset_list[1]))
