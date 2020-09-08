from config import autoconfigure
import numpy as np
import torchvision
from skimage.transform import resize
from torch.utils.data import Dataset, random_split
import os
from PIL import Image
import torch
from datetime import datetime

autoconfigure()
width = 128
height = 256
nb_classes = 5
nb_anchors = 4  # one per foreground class
input_channels = 61
base_channels = 64
batch_size = 618

pad_left_range = 0.2
pad_top_range = 0.2
pad_right_range = 0.2
pad_bot_range = 0.2
dir_np_chargrid_1h = os.getenv('DIR_NP_CHARGRID_1H')
dir_np_gt_1h = os.getenv('DIR_NP_GT_1H')
dir_np_bbox_anchor_mask = os.getenv('DIR_NP_BBOX_ANCHOR_MASK')
dir_np_bbox_anchor_coord = os.getenv('DIR_NP_BBOX_ANCHOR_COORD')
list_filenames = [f for f in os.listdir(dir_np_chargrid_1h) if os.path.isfile(os.path.join(dir_np_chargrid_1h, f))]
# list_filenames = list_filenames[:10]


def augment_data(data, tab_rand, order, shape, coord=False):
    data_temp = resize(np.pad(data, ((tab_rand[1], tab_rand[3]), (tab_rand[0], tab_rand[2]), (0, 0)), 'constant'),
                       shape, order=order, anti_aliasing=True)

    if coord:
        for i in range(0, nb_anchors):
            mask = (data_temp > 1e-6)[:, :, 4 * i]
            data_temp[mask, 4 * i] *= shape[1]
            data_temp[mask, 4 * i] += tab_rand[0]
            data_temp[mask, 4 * i] /= (tab_rand[0] + shape[1] + tab_rand[2])

            data_temp[mask, 4 * i + 2] *= shape[1]
            data_temp[mask, 4 * i + 2] += tab_rand[0]
            data_temp[mask, 4 * i + 2] /= (tab_rand[0] + shape[1] + tab_rand[2])

            data_temp[mask, 4 * i + 1] *= shape[0]
            data_temp[mask, 4 * i + 1] += tab_rand[1]
            data_temp[mask, 4 * i + 1] /= (tab_rand[1] + shape[0] + tab_rand[3])

            data_temp[mask, 4 * i + 3] *= shape[0]
            data_temp[mask, 4 * i + 3] += tab_rand[1]
            data_temp[mask, 4 * i + 3] /= (tab_rand[1] + shape[0] + tab_rand[3])

    return data_temp


def extract_combined_data(dataset, batch_size, pad_left_range, pad_top_range, pad_right_range, pad_bot_range):
    if batch_size > len(dataset):
        raise ValueError('batch_size > length of dataset {}'.format(len(dataset)))

    np.random.shuffle(dataset)
    tab_rand = np.random.rand(batch_size, 4) * [pad_left_range * width, pad_top_range * height, pad_right_range * width,
                                                pad_bot_range * height]

    tab_rand = tab_rand.astype(int)
    chargrid_input, seg_gt, anchor_mask_gt, anchor_coord_gt = [], [], [], []

    for i in range(0, batch_size):
        data = np.load(os.path.join(dir_np_chargrid_1h, dataset[i]))
        chargrid_input.append(data)
        # chargrid_input.append(augment_data(data, tab_rand[i], order=1, shape=(height, width, input_channels)))

        data = np.load(os.path.join(dir_np_gt_1h, dataset[i]))
        seg_gt.append(data)
        # seg_gt.append(augment_data(data, tab_rand[i], order=1, shape=(height, width, nb_classes)))

        data = np.load(os.path.join(dir_np_bbox_anchor_mask, dataset[i]))
        anchor_mask_gt.append(data)
        # anchor_mask_gt.append(augment_data(data, tab_rand[i], order=1, shape=(height, width, 2 * nb_anchors)))

        data = np.load(os.path.join(dir_np_bbox_anchor_coord, dataset[i]))
        anchor_coord_gt.append(data)
        # anchor_coord_gt.append(
        #     augment_data(data, tab_rand[i], order=0, shape=(height, width, 4 * nb_anchors), coord=True))

    return np.array(chargrid_input), np.array(seg_gt), np.array(anchor_mask_gt), np.array(anchor_coord_gt)


time_then = datetime.now()
# print(time_then)

# Extract combined data here
chargrid_input, seg_gt, anchor_mask_gt, anchor_coord = extract_combined_data(list_filenames, batch_size, pad_left_range,
                                                                             pad_top_range, pad_right_range,
                                                                             pad_bot_range)

print("total time taken for file parsing: ")
print((datetime.now() - time_then).total_seconds())

class ChargridDataset(Dataset):
    def __init__(self, chargrid_input, segmentation_ground_truth, anchor_mask_ground_truth, anchor_coordinates):
        self.chargrid_input = chargrid_input
        self.segmentation_ground_truth = segmentation_ground_truth
        self.anchor_mask_ground_truth = anchor_mask_ground_truth
        self.anchor_coordinates = anchor_coordinates
        return

    def __len__(self):
        return len(self.chargrid_input)

    def __getitem__(self, idx):
        if type(idx) is torch.Tensor:
            index = idx.item()

        segmentation_label = self.segmentation_ground_truth[idx]
        anchor_mask_label = self.anchor_mask_ground_truth[idx]
        anchor_coordinates_label = self.anchor_coordinates[idx]
        image = Image.fromarray(np.uint8(np.squeeze(self.chargrid_input[idx, :, :, 0:3] * 255)))

        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        return transforms(image), transforms(segmentation_label), \
               transforms(anchor_mask_label), transforms(anchor_coordinates_label)


def get_dataset():
    dataset = ChargridDataset(chargrid_input, seg_gt, anchor_mask_gt, anchor_coord)
    # print('Dataset length is {0}'.format(len(dataset)))
    test_no = int(len(dataset) * 0.2)
    trainset, testset = random_split(dataset, [len(dataset) - test_no, test_no])

    # print(len(trainset), len(testset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True, num_workers=0)

    return trainloader, testloader


if __name__ == '__main__':

    print('calling get_dataset')
    trainloader, testloader = get_dataset()
    img, l1, l2, l3 = next(iter(trainloader))

    print(img.shape, l1.shape, l2.shape, l3.shape)
