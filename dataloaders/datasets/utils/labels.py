# -*- coding:utf-8 -*-
#
# rssrai2019 labels
#
from collections import namedtuple

# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple('Label', [
    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
             # We use them to uniquely name a class
    'id',  # An integer ID that is associated with this label.
           # The IDs are used to represent the label in ground truth images
           # An ID of -1 means that this label does not have an ID and thus
           # is ignored when creating ground truth images (e.g. license plate).
           # Do not modify these IDs, since exactly these IDs are expected by the
           # evaluation server.
    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
                # ground truth images with train IDs, using the tools provided in the
                # 'preparation' folder. However, make sure to validate or submit results
                # to our evaluation server using the regular IDs above!
                # For trainIds, multiple labels might have the same ID. Then, these labels
                # are mapped to the same class in the ground truth images. For the inverse
                # mapping, we use the label that is defined first in the list below.
                # For example, mapping all void-type classes to the same ID in training,
                # might make sense for some approaches.
                # Max value is 255!
    'category',  # The name of the category that this label belongs to
    'categoryId',  # The ID of this category. Used to create ground truth images
                   # on category level.
    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
                     # during evaluations or not
    'color',  # The color of this label
    ])


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name    id   trainId   category       catId     ignoreInEval   color
    Label('其他类别',  0,   0,       'void',         0,       False,         (0, 0, 0)),

    Label('水   田',  1,   1,       'farmland',     1,       False,        (0, 200, 0)),
    Label('水 浇地',  2,   2,        'farmland',    1,       False,        (150, 250, 0)),
    Label('旱 耕地',  3,   3,        'farmland',    1,       False,        (150, 200, 150)),

    Label('园   地',  4,   4,       'woodland',     2,       False,        (200, 0, 200)),
    Label('乔木林地',  5,   5,       'woodland',     2,       False,       (150, 0, 250)),
    Label('灌木林地',  6,   6,       'woodland',     2,       False,       (150, 150, 250)),

    Label('天然草地',  7,   7,       'grassland',    3,       False,       (250, 200, 0)),
    Label('人工草地',  8,   8,       'grassland',    3,       False,       (200, 200, 0)),

    Label('工业用地',  9,   9,       'urbanland',    4,       False,       (200, 0, 0)),
    Label('城市住宅', 10,   10,      'urbanland',    4,       False,       (250, 0, 150)),
    Label('村镇住宅', 11,   11,      'urbanland',    4,       False,       (200, 150, 150)),
    Label('交通运输', 12,   12,      'urbanland',    4,       False,       (250, 150, 150)),

    Label('河   流', 13,   13,      'waterland',    5,       False,        (0, 0, 200)),
    Label('湖   泊', 14,   14,      'waterland',    5,       False,        (0, 150, 200)),
    Label('坑   塘', 15,   15,      'waterland',    5,       False,        (0, 200, 250)),
]


# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# color to label object
color2label = {label.color: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]


# --------------------------------------------------------------------------------
# Main for testing
# --------------------------------------------------------------------------------


if __name__ == "__main__":
    # Print all the labels
    print("List of rssrai2019 labels:")
    print("")
    print("{:>15} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12}".format('name', 'id', 'trainId', 'category', 'categoryId',
                                                                     'ignoreInEval'))
    print("    " + ('-' * 98))
    for label in labels:
        print("{:>15} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12}".format(label.name, label.id, label.trainId,
                                                                         label.category, label.categoryId,
                                                                         label.ignoreInEval))
    print("")

    print("Example usages:")

    # Map from name to label
    name = '水   田'
    id = name2label[name].id
    print("ID of label '{name}': {id}".format(name=name, id=id))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format(id=id, category=category))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format(id=trainId, name=name))
