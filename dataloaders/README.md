# 数据集处理步骤
1. 在labels.py中定义好所有类别，及其对应的颜色

2. 修改createTrainIdLabelImgs.py中color标注图片所在的路径，然后运行该脚本，生成color图片对应的train_id图片。便于模型训练时，使用该图片进行判定。

3. 修改cropTIFFImg.py中各图片路径，和crop_size大小，对所有图片进行裁剪。模型训练时，使用裁剪后的图片训练。