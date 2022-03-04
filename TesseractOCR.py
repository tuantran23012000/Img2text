import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import re
import sys
from tqdm import tqdm
from autocorrect import Speller
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = './weights/transformerocr.pth'
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
detector = Predictor(config)
spell = Speller(lang='vi')
#print(spell("Xin chà, tôi nà Tuấn"))
patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}

def convert(text):
    """
    Convert from 'Tieng Viet co dau' thanh 'Tieng Viet khong dau'
    text: input string to be converted
    Return: string converted
    """
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        # deal with upper case
        output = re.sub(regex.upper(), replace.upper(), output)
    return output

# Path of working folder on Disk
src_path = "E:/Tuan/Image-to-Text-converter-OCR/"
pytesseract.pytesseract.tesseract_cmd = 'E:/tesseract/tesseract.exe'
def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite(src_path + "removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image after apply opencv to do some ...
    cv2.imwrite(src_path + "thres.png", img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(Image.open(src_path + "thres.png"),lang="vie")
    result = result.split("\n")
    print(result[0])
    # Remove template file
    #os.remove(temp)

    return result
def get_string1(img_path):
    # Read image using opencv
    img = cv2.imread(img_path)
   # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]
    # Create a directory for outputs
    output_path = os.path.join('output_path', "ocr")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Converting to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Removing Shadows
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    
    #Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)#increases the white region in the image 
    img = cv2.erode(img, kernel, iterations=1) #erodes away the boundaries of foreground object
    
    #Apply blur to smooth out the edges
    #img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    #Save the filtered image in the output directory
    save_path = os.path.join(output_path, file_name + "_filter_" + str('as') + ".jpg")
    cv2.imwrite(save_path, img)
    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="eng")
    
    print(result)
    return result
def get_string2(img):
    # Read image with opencv
    #img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite(src_path + "removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image after apply opencv to do some ...
    cv2.imwrite(src_path + "thres.png", img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(Image.open(src_path + "thres.png"),lang="vie")
    if result[0]!='\x0c':
        result = result.split("\n")
        #result = spell(result[0])
        result = result[0]
        result = convert(result)
        result = result.lower()
        #print(result)
        #print(result[0])
        return result
    else:
        result=""
        return result
def getstring3(img):
    # # Convert to gray
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Apply dilation and erosion to remove some noise
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1)

    # # Write image after removed noise
    # cv2.imwrite(src_path + "removed_noise.png", img)

    # #  Apply threshold to get image with only black and white
    # #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # # Write the image after apply opencv to do some ...
    # cv2.imwrite(src_path + "thres.png", img)
    # #plt.imshow(img)
    # #result = pytesseract.image_to_string(Image.open(src_path + "thres.png"),lang="vie")
    # s = detector.predict(Image.open(src_path + "thres.png"))
    s = str(detector.predict(img))
    result = convert(s)
    result = result.lower()
    result = result.replace(".",",")
    #print(result)
    if ("bui khuong duy" in result or "vnd" in result or "15,000" in result or "20,000" in result or "30,000" in result) and "o vnd" not in result:
        return result
    else: return ""
data_path = "E:/Tuan/DBNet/datasets/test/output/check"
file_names = []
inf_all = []
# print(pd.read_csv("framingham.csv"))
tmp=0
df = pd.read_csv("framingham.csv")
for folder in tqdm(os.listdir(data_path)):
    tmp+=1
    folder_p = os.path.join(data_path,folder)
    file_names.append(folder)
    inf = []
    for img_ in os.listdir(folder_p):

        #img = cv2.imread(os.path.join(folder_p,img_))
        img = Image.open(os.path.join(folder_p,img_))
        #df = pd.DataFrame(data, columns = ['Name', 'Age'])
        if img is not None:
            r = getstring3(img)
            #print(r)
            if len(r)!=0 and r not in inf:
                inf.append(str(r))
    if len(inf)!=0:
        inf_all.append(inf)
    else:
        inf_all.append("không có thông tin")
        # else:
        #     print(img,os.path.join(data_path,img_))
    # if tmp==5:
    #     break
df["Filenames"] = file_names
df["Money"] = inf_all
print(df)
df.to_csv("inf.csv", encoding='utf-8', index=False)

