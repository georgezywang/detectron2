#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:06:09 2020

@author: wzy
"""
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image
import json
import datetime
import os

class ImageLabel():
    def __init__(self, name, mask, cat, id = None, crowd = False):
        self.name = name
        self.mask= mask
        self.cat= cat
        self.id = id
        self.crowd = crowd

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        elif isinstance(obj, datetime.timedelta):
            return (datetime.datetime.min + obj).time().isoformat()
        return super(DateTimeEncoder, self).default(obj)

class MaskParser():
    def __init__(self, catDic, imgList, inPath = "./", description = \
                 "Dataset with Mask", url = "", contributor = "", \
                     version = "1", outPath = "data.json", useImgID = False):
        assert type(catDic) == dict, "Category Dictionary Must be a dictionary"
        self.catDic = catDic
        try:
            imgList[0]
        except:
            print("image list not a list of is emty")
        assert isinstance(imgList[0], ImageLabel), "list items must be ImageLabel"
        self.imgList = imgList
        self.description = description
        self.inPath = inPath
        self.url = url
        self.contributor = contributor
        self.version = version
        self.license = []
        self.cat = []
        self.info = None
        self.img = []
        self.annotations = []
        self.json = None
        self.outPath = outPath
        assert type(useImgID) == bool, "Use Image ID must be boolean"
        self.useImgID = useImgID
        
        self.initInfo()
        self.initCat()
        self.addLic()
        self.createMaskList()
        self.composeJson()
        
    def initInfo(self):
        info = {
            "year": int(datetime.datetime.now().year),
            "version": self.version,
            "description": self.description,
            "contributor": self.contributor,
            "url": self.url,
            "date_created": datetime.datetime.now()
        }
        info = DateTimeEncoder().encode(info)
        self.info = json.loads(info)
    
    def addLic(self, id = 1, url = "", name = ""):
        lic = {
            "id": id,
            "url": url,
            "name": name
        }
        self.license.append(lic)
    
    def initCat(self):
        for key in self.catDic:
            cat = {
            "id": self.catDic[key],
            "name": key,
            "supercategory": "none"
            }
            self.cat.append(cat)
            
    def createMaskList(self):
        iter = 0
        for currImg in self.imgList:
            path = self.inPath
            mask = currImg.mask
            img = currImg.name
            maskPath = os.path.join(path, mask)
            #imgPath = os.path.join(path, img)
            
            try:
                maskImg = Image.open(maskPath)
            except:
                print("image " + maskPath + "does not exist")
                continue;
            
            if(self.useImgID == False):
                id = iter
            else:
                id = currImg.id
            
            width, height = maskImg.size
            assert width > 0 and height > 0, "empty image"
            if(type(maskImg.getpixel((0,0))) != int):
                maskImg = self.redDim(maskImg)
            
            img = {
              "id": id,
              "license": "",
              "file_name": img,
              "height": height,
              "width": width,
              "date_captured": datetime.datetime.now()
              }
            img = DateTimeEncoder().encode(img)
            
            self.img.append(json.loads(img))
            ann = self.createSubMaskAnnotation(maskImg, id, currImg.cat, id, currImg.crowd)
            self.annotations.append(ann)
            iter = iter + 1
    
    def composeJson(self):
        rec = {
           "info": self.info,
           "licenses": self.license,
           "categories": self.cat,
           "images": self.img,
           "annotations": self.annotations
           }
        self.json = rec
    
    def saveJson(self):
        with open(self.outPath, 'w') as json_file:
            json.dump(self.json, json_file)
    
    def redDim(self, maskImg):
        width, height = maskImg.size
        blankImg = np.zeros((height, width))
        for i in range(width):
            for j in range(height):
                if(maskImg.getpixel((i,j)) != (0, 0, 0)):
                    blankImg[j,i] = 1
        return blankImg
                
        
    def createSubMaskAnnotation(self, sub_mask, image_id, category_id, annotation_id, is_crowd):
        contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
        print("Creating annotations for image " + str(image_id))
        segmentations = []
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)
    
            # Make a polygon and simplify it
            poly = Polygon(contour)
            try:
                poly = poly.simplify(1.0, preserve_topology=False)
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                segmentations.append(segmentation)
                polygons.append(poly)
            except:
                polyIndex = 0
                while(True):
                    try:
                        part = poly[polyIndex].simplify(1.0, preserve_topology=False)
                        segmentation = np.array(part.exterior.coords).ravel().tolist()
                        segmentations.append(segmentation)
                        polygons.append(part)
                    except:
                        break
                    polyIndex = polyIndex + 1
    
        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area
    
        annotation = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
        }
    
        return annotation
