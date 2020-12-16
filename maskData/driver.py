import MaskCoco
import os

def fileDic(path):
    imgList = []
    for i in range(501):
        mask = str(i).zfill(3) + "_gt.png"
        img = str(i).zfill(3) + "_rgb.png"
        maskPath = os.path.join(path, mask)
        if os.path.exists(maskPath):
            img = MaskCoco.ImageLabel(img, mask, 0, i)
            imgList.append(img)
    return imgList

catDic = {"fire": 0}
"""

imgList = fileDic("fire/train")
myParser = MaskCoco.MaskParser(catDic, imgList, inPath = "fire/train",\
                               outPath="fire/train/dat.json", useImgID = True)
myParser.saveJson()


imgList = fileDic("fire/val")
myParser = MaskCoco.MaskParser(catDic, imgList, inPath = "fire/val",\
                               outPath="fire/val/dat.json", useImgID = True)
myParser.saveJson()
"""

imgList = fileDic("fire/test")
myParser = MaskCoco.MaskParser(catDic, imgList, inPath = "fire/test",\
                               outPath="fire/test/dat.json", useImgID = True)
myParser.saveJson()
