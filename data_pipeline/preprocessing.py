# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 25/03/2020
"""
First preprocessing program that :
- generates Chargrids from input images thanks to Tesseract
- extracts bounding boxes for each class from the ground truth files
- generates class segmentation from the class bounding boxes
- reduces the size of images by removing empty rows and empty columns

Requirements
----------
- Tesseract must be installed in "C:\Program Files\Tesseract-OCR/tesseract"
- Input images must be located in the folder dir_img = "./data/img_inputs/"
- Input bounding boxes (ground truth) must be located in the folder dir_boxes = "./data/gt_boxes/"
- Input classes (ground truth) must be located in the folder dir_classes = "./data/gt_classes/"

Hyperparameters
----------
- tesseract_conf_threshold : gives a threshold below which the tesseract information is not kept
- cosine_similarity_threshold : gives a threshold above which two strings are considered similar

Return
----------
Several files are generated :
- in outdir_np_chargrid = "./data/np_chargrids/" : Chargrids of each input image in npy (numpy array format)
- in outdir_png_chargrid = "./data/img_chargrids/" : Chargrids of each input image in png
- in outdir_np_gt = "./data/np_gt/" : Class Segmentation of each input image in npy (numpy array format)
- in outdir_png_gt = "./data/img_gt/" : Class Segmentation of each input image in png
- in outdir_pd_bbox = "./data/pd_bbox/" : Class Bounding Boxes of each input image in pkl (pandas dataframe format)
"""
# suppress Pandas future warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract as te
import os
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import autoconfigure
from alive_progress import alive_bar

autoconfigure()
te.pytesseract.tesseract_cmd = os.getenv("TESSERACT_EXECUTABLE")

## Hyperparameters
dir_img = os.getenv("DIR_IMG")
dir_boxes = os.getenv("DIR_BOXES")
dir_classes = os.getenv("DIR_CLASSES")

outdir_np_chargrid = os.getenv("DIR_NP_CHARGRID")
outdir_png_chargrid = os.getenv("DIR_PNG_CHARGRID")
outdir_np_gt = os.getenv("DIR_NP_GT")
outdir_png_gt = os.getenv("DIR_PNG_GT")
outdir_pd_bbox = os.getenv("DIR_PD_BBOX")

tesseract_conf_threshold = 10
cosine_similarity_threshold = 0.4
list_classes = ["total", "company", "address", "date"]
nb_classes = len(list_classes)


def add_row_gt_pd(row, c, gt_pd):
    return gt_pd.append(
        {
            "left": row["top_left_x"],
            "top": row["top_left_y"],
            "right": row["bot_right_x"],
            "bot": row["bot_right_y"],
            "class": c,
        },
        ignore_index=True,
    )


def extract_tesseract_information(filename):
    # logger.info('input ->' + os.path.join(dir_img, filename))
    img = plt.imread(os.path.join(dir_img, filename), format="jpeg")
    # print(filename, img.shape)

    dt = te.image_to_data(
        img, config="", output_type=te.Output.DATAFRAME, pandas_config=None
    )
    dt = dt[dt["conf"] > tesseract_conf_threshold]
    dt["text"] = dt["text"].astype("str")

    return dt, img.shape


def get_chargrid(dt, img_shape):
    chargrid_pd = pd.DataFrame(
        columns=["left", "top", "width", "height", "ord", "conf"]
    )

    for index, row in dt.iterrows():
        for i in range(0, len(row["text"])):
            row["width"] = (
                (row["width"] + len(row["text"]) - 1)
                // len(row["text"])
                * len(row["text"])
            )  # Split character by character

            chargrid_pd = chargrid_pd.append(
                {
                    "left": row["left"] + row["width"] * i // len(row["text"]),
                    "top": row["top"],
                    "width": row["width"] // len(row["text"]),
                    "height": row["height"],
                    "ord": ord(row["text"][i]),
                    "conf": row["conf"],
                },
                ignore_index=True,
            )

    chargrid_pd = chargrid_pd[chargrid_pd["ord"] >= 33]
    chargrid_pd = chargrid_pd[chargrid_pd["ord"] <= 126]
    chargrid_pd["ord"] -= 32

    chargrid_np = np.array([0] * img_shape[0] * img_shape[1]).reshape(
        (img_shape[0], img_shape[1])
    )

    for index, row in chargrid_pd.iterrows():
        # Assign text unicode code in chargrid_np
        # for example--- ord('A') = 65

        # suppose first character is 'A' in chargrid_pd row
        # and we have top, left, width and height of occurence of 'A' in doc

        # so place the value '65' at particualr position in chargrid_np calculated by using top, left, width, height

        chargrid_np[
            int(row["top"]) : int(row["top"]) + int(row["height"]),
            int(row["left"]) : int(row["left"]) + int(row["width"]),
        ] = int(row["ord"])

        # if top-3, height - 5, left-1, width-10
        # then chargrid_np[3: 8, 1:11]= 65
        # that means 3 to 8 row and 1 to 11 columns should be occupied by 65

    return chargrid_np


def get_groundTruth(filename):
    gt_pd = pd.DataFrame(columns=["left", "top", "right", "bot", "class"])

    ## Import ground truth files
    # create pd_boxes empty dataframe with the similar column names in dir_boxes

    pd_boxes = pd.DataFrame(
        columns=[
            "top_left_x",
            "top_left_y",
            "top_right_x",
            "top_right_y",
            "bot_left_x",
            "bot_left_y",
            "bot_right_x",
            "bot_right_y",
            "text",
        ]
    )
    dic_class = dict()

    with open(os.path.join(dir_boxes, filename).replace("jpg", "txt")) as f:
        reader = f.read().splitlines()
        pd_boxes = pd.DataFrame(
            [x.split(",", 8) for x in reader],
            columns=[
                "top_left_x",
                "top_left_y",
                "top_right_x",
                "top_right_y",
                "bot_right_x",
                "bot_right_y",
                "bot_left_x",
                "bot_left_y",
                "text",
            ],
        )

        pd_boxes["top_left_x"] = pd_boxes["top_left_x"].astype("int")
        pd_boxes["top_left_y"] = pd_boxes["top_left_y"].astype("int")
        pd_boxes["top_right_x"] = pd_boxes["top_right_x"].astype("int")
        pd_boxes["top_right_y"] = pd_boxes["top_right_y"].astype("int")
        pd_boxes["bot_left_x"] = pd_boxes["bot_left_x"].astype("int")
        pd_boxes["bot_left_y"] = pd_boxes["bot_left_y"].astype("int")
        pd_boxes["bot_right_x"] = pd_boxes["bot_right_x"].astype("int")
        pd_boxes["bot_right_y"] = pd_boxes["bot_right_y"].astype("int")
        pd_boxes["text"] = pd_boxes["text"].str.upper()

    # import dir_classes, here nb_classes = 4, list_classes is given above
    with open(os.path.join(dir_classes, filename).replace("jpg", "json")) as f:
        dic_class = json.load(f)

    for i in range(nb_classes):
        if list_classes[i] not in dic_class.keys():
            # if not a single class name matched with dic_class.keys() then make it "UNKNOWN"
            dic_class[list_classes[i]] = "UNKNOWN"

        dic_class[list_classes[i]] = dic_class[list_classes[i]].upper()

    # Detect classes in the bounding box file
    # label coordinates with class index

    vectorized_text = CountVectorizer().fit_transform(
        [dic_class[list_classes[i]] for i in range(nb_classes)]
        + pd_boxes["text"].tolist()
    )

    for index, row in pd_boxes.iterrows():

        # Classes of type string
        # print(vectorized_text.toarray())

        # use cosine similarity to check relation between text and classes
        # if cosine_similarity greater than threshold then text must belong to that class
        # if angle is small, it means high similarity

        # if string---must be company, address
        # if float---must be total
        # if date type-- must be date

        if (
            cosine_similarity(
                vectorized_text[0].reshape(1, -1),
                vectorized_text[index + nb_classes].reshape(1, -1),
            )[0][0]
            > cosine_similarity_threshold
        ):
            gt_pd = add_row_gt_pd(row, 3, gt_pd)

        if (
            cosine_similarity(
                vectorized_text[2].reshape(1, -1),
                vectorized_text[index + nb_classes].reshape(1, -1),
            )[0][0]
            > cosine_similarity_threshold
        ):
            gt_pd = add_row_gt_pd(row, 2, gt_pd)

        # Classes of type date
        tab_date = re.findall(
            r"((?i)(?:[12][0-9]|3[01]|0*[1-9])(?P<sep>[- \/.\\])(?P=sep)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep)+(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?P<sep2>[- \/.\\])(?P=sep2)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep2)+\d\d|(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P<sep3>[- \/.\\])(?P=sep3)*(?:[12][0-9]|3[01]|0*[1-9])(?P=sep3)+(?:19|20)\d\d|(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P<sep4>[- \/.\\])(?P=sep4)*(?:[12][0-9]|3[01]|0*[1-9])(?P=sep4)+\d\d|(?:19|20)\d\d(?P<sep5>[- \/.\\])(?P=sep5)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep5)+(?:[12][0-9]|3[01]|0*[1-9])|\d\d(?P<sep6>[- \/.\\])(?P=sep6)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep6)+(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])(?:19|20)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])\d\d|(?:19|20)\d\d(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|\d\d(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0[1-9])(?:1[012]|0[1-9])(?:19|20)\d\d|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])(?:19|20)\d\d|(?:19|20)\d\d(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])\d\d|(?:[12][0-9]|3[01]|0[1-9])(?:1[012]|0[1-9])\d\d|\d\d(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9]))",
            row["text"],
        )
        for dat in tab_date:
            if dat[0] == dic_class["date"]:
                gt_pd = add_row_gt_pd(row, 4, gt_pd)

        # Classes of type float
        tab_floats = re.findall(r"([-+]?[0-9]*\.?[0-9]+)", row["text"])
        total_float = re.search(r"([-+]?[0-9]*\.?[0-9]+)", dic_class["total"])
        if total_float:
            for flo in tab_floats:
                if float(total_float.group(0)) == float(flo):
                    gt_pd = add_row_gt_pd(row, 1, gt_pd)

    return gt_pd


def get_final_groundtruth(gt_pd, chargrid_np, img_shape):
    gt_np = np.array([0] * img_shape[0] * img_shape[1]).reshape(
        (img_shape[0], img_shape[1])
    )

    gt_pd.sort_values(by="class", ascending=True, inplace=True)  # Sort by confidence
    gt_pd.reset_index(drop=True, inplace=True)

    for index, row in gt_pd.iterrows():
        gt_np[row["top"] : row["bot"], row["left"] : row["right"]] = row["class"]

    ## Remove empty rows and columns
    tab_cumsum_todelete_x = np.cumsum(np.all(chargrid_np == 0, axis=0))
    gt_pd["left"] -= tab_cumsum_todelete_x[gt_pd["left"].tolist()]
    gt_pd["right"] -= tab_cumsum_todelete_x[gt_pd["right"].tolist()]

    tab_cumsum_todelete_y = np.cumsum(np.all(chargrid_np == 0, axis=1))
    gt_pd["top"] -= tab_cumsum_todelete_y[gt_pd["top"].tolist()]
    gt_pd["bot"] -= tab_cumsum_todelete_y[gt_pd["bot"].tolist()]

    gt_np = gt_np[:, ~np.all(chargrid_np == 0, axis=0)]
    gt_np = gt_np[~np.all(chargrid_np == 0, axis=1), :]

    chargrid_np = chargrid_np[:, ~np.all(chargrid_np == 0, axis=0)]
    chargrid_np = chargrid_np[~np.all(chargrid_np == 0, axis=1), :]

    return gt_pd, gt_np, chargrid_np


#################################################################################################################


def extract_class_bounding_boxes(filename):
    gt_pd = pd.DataFrame(columns=["left", "top", "right", "bot", "class"])

    ## Import ground truth files
    pd_boxes = pd.DataFrame(
        columns=[
            "top_left_x",
            "top_left_y",
            "top_right_x",
            "top_right_y",
            "bot_left_x",
            "bot_left_y",
            "bot_right_x",
            "bot_right_y",
            "text",
        ]
    )
    dic_class = dict()

    with open(os.path.join(dir_boxes, filename).replace("jpg", "txt")) as f:
        reader = f.read().splitlines()
        pd_boxes = pd.DataFrame(
            [x.split(",", 8) for x in reader],
            columns=[
                "top_left_x",
                "top_left_y",
                "top_right_x",
                "top_right_y",
                "bot_right_x",
                "bot_right_y",
                "bot_left_x",
                "bot_left_y",
                "text",
            ],
        )

        pd_boxes["top_left_x"] = pd_boxes["top_left_x"].astype("int")
        pd_boxes["top_left_y"] = pd_boxes["top_left_y"].astype("int")
        pd_boxes["top_right_x"] = pd_boxes["top_right_x"].astype("int")
        pd_boxes["top_right_y"] = pd_boxes["top_right_y"].astype("int")
        pd_boxes["bot_left_x"] = pd_boxes["bot_left_x"].astype("int")
        pd_boxes["bot_left_y"] = pd_boxes["bot_left_y"].astype("int")
        pd_boxes["bot_right_x"] = pd_boxes["bot_right_x"].astype("int")
        pd_boxes["bot_right_y"] = pd_boxes["bot_right_y"].astype("int")
        pd_boxes["text"] = pd_boxes["text"].str.upper()

    with open(os.path.join(dir_classes, filename).replace("jpg", "json")) as f:
        dic_class = json.load(f)
    for i in range(nb_classes):
        if list_classes[i] not in dic_class.keys():
            dic_class[list_classes[i]] = "UNKNOWN"
        dic_class[list_classes[i]] = dic_class[list_classes[i]].upper()

    ## Detect classes in the bounding box file
    vectorized_text = CountVectorizer().fit_transform(
        [dic_class[list_classes[i]] for i in range(nb_classes)]
        + pd_boxes["text"].tolist()
    )

    for index, row in pd_boxes.iterrows():
        # Classes of type string
        # print(vectorized_text.toarray())
        if (
            cosine_similarity(
                vectorized_text[0].reshape(1, -1),
                vectorized_text[index + nb_classes].reshape(1, -1),
            )[0][0]
            > cosine_similarity_threshold
        ):
            gt_pd = add_row_gt_pd(row, 3, gt_pd)

        if (
            cosine_similarity(
                vectorized_text[2].reshape(1, -1),
                vectorized_text[index + nb_classes].reshape(1, -1),
            )[0][0]
            > cosine_similarity_threshold
        ):
            gt_pd = add_row_gt_pd(row, 2, gt_pd)

        # Classes of type date
        tab_date = re.findall(
            r"((?i)(?:[12][0-9]|3[01]|0*[1-9])(?P<sep>[- \/.\\])(?P=sep)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep)+(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?P<sep2>[- \/.\\])(?P=sep2)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep2)+\d\d|(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P<sep3>[- \/.\\])(?P=sep3)*(?:[12][0-9]|3[01]|0*[1-9])(?P=sep3)+(?:19|20)\d\d|(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P<sep4>[- \/.\\])(?P=sep4)*(?:[12][0-9]|3[01]|0*[1-9])(?P=sep4)+\d\d|(?:19|20)\d\d(?P<sep5>[- \/.\\])(?P=sep5)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep5)+(?:[12][0-9]|3[01]|0*[1-9])|\d\d(?P<sep6>[- \/.\\])(?P=sep6)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep6)+(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])(?:19|20)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])\d\d|(?:19|20)\d\d(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|\d\d(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0[1-9])(?:1[012]|0[1-9])(?:19|20)\d\d|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])(?:19|20)\d\d|(?:19|20)\d\d(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])\d\d|(?:[12][0-9]|3[01]|0[1-9])(?:1[012]|0[1-9])\d\d|\d\d(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9]))",
            row["text"],
        )
        for dat in tab_date:
            if dat[0] == dic_class["date"]:
                gt_pd = add_row_gt_pd(row, 4, gt_pd)

        # Classes of type float
        tab_floats = re.findall(r"([-+]?[0-9]*\.?[0-9]+)", row["text"])
        total_float = re.search(r"([-+]?[0-9]*\.?[0-9]+)", dic_class["total"])
        if total_float:
            for flo in tab_floats:
                if float(total_float.group(0)) == float(flo):
                    gt_pd = add_row_gt_pd(row, 1, gt_pd)

    return gt_pd


def plot_input_vs_output(input, output):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(input)
    ax2.imshow(output)
    plt.show()
    plt.clf()


def get_reduced_output(chargrid_pd, gt_pd, img_shape):
    chargrid_np = np.array([0] * img_shape[0] * img_shape[1]).reshape(
        (img_shape[0], img_shape[1])
    )

    chargrid_pd.sort_values(
        by="conf", ascending=True, inplace=True
    )  # Sort by confidence
    chargrid_pd.reset_index(drop=True, inplace=True)

    for index, row in chargrid_pd.iterrows():
        chargrid_np[
            row["top"] : row["top"] + row["height"],
            row["left"] : row["left"] + row["width"],
        ] = row["ord"]

    gt_np = np.array([0] * img_shape[0] * img_shape[1]).reshape(
        (img_shape[0], img_shape[1])
    )

    gt_pd.sort_values(by="class", ascending=True, inplace=True)  # Sort by confidence
    gt_pd.reset_index(drop=True, inplace=True)

    for index, row in gt_pd.iterrows():
        gt_np[row["top"] : row["bot"], row["left"] : row["right"]] = row["class"]

    ## Remove empty rows and columns
    tab_cumsum_todelete_x = np.cumsum(np.all(chargrid_np == 0, axis=0))
    gt_pd["left"] -= tab_cumsum_todelete_x[gt_pd["left"].tolist()]
    gt_pd["right"] -= tab_cumsum_todelete_x[gt_pd["right"].tolist()]

    tab_cumsum_todelete_y = np.cumsum(np.all(chargrid_np == 0, axis=1))
    gt_pd["top"] -= tab_cumsum_todelete_y[gt_pd["top"].tolist()]
    gt_pd["bot"] -= tab_cumsum_todelete_y[gt_pd["bot"].tolist()]

    gt_np = gt_np[:, ~np.all(chargrid_np == 0, axis=0)]
    gt_np = gt_np[~np.all(chargrid_np == 0, axis=1), :]

    chargrid_np = chargrid_np[:, ~np.all(chargrid_np == 0, axis=0)]
    chargrid_np = chargrid_np[~np.all(chargrid_np == 0, axis=1), :]

    return chargrid_np, gt_np, gt_pd


if __name__ == "__main__":
    list_filenames = [
        f for f in os.listdir(dir_img) if os.path.isfile(os.path.join(dir_img, f))
    ]

    logger.info("Number of input files : " + str(len(list_filenames)))

    with alive_bar(
        len(list_filenames), ctrl_c=False, title=f"processing files: ", bar="classic"
    ) as bar:
        for filename in list_filenames:
            df, img_shape = extract_tesseract_information(filename)

            chargrid_np = get_chargrid(df, img_shape)

            gt_pd = get_groundTruth(filename)

            gt_pd, gt_np, chargrid_np = get_final_groundtruth(
                gt_pd, chargrid_np, img_shape
            )

            plot_input_vs_output(chargrid_np, gt_np)
            # print(gt_pd)

            ##Saving
            np.save(
                os.path.join(outdir_np_chargrid, filename).replace("jpg", "npy"),
                chargrid_np,
            )
            np.save(os.path.join(outdir_np_gt, filename).replace("jpg", "npy"), gt_np)
            gt_pd.to_pickle(
                os.path.join(outdir_pd_bbox, filename).replace("jpg", "pkl")
            )

            plt.imshow(chargrid_np)
            plt.savefig(
                os.path.join(outdir_png_chargrid, filename).replace("jpg", "png")
            )
            plt.close()

            plt.imshow(gt_np)
            plt.savefig(os.path.join(outdir_png_gt, filename).replace("jpg", "png"))
            plt.close()
            bar()
    print("processing complete")
