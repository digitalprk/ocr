import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image 
import random
import copy
import numpy
import os
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from inspect import getsourcefile
from os.path import abspath
import unidecode

path = os.path.dirname(abspath(getsourcefile(lambda:0)))

def elastic_transform(image, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       https://gist.github.com/fmder/e28813c1e8721830ff9c
    """
    
    alpha = random.randint(7, 17)
    sigma = random.randint(2, 5) if alpha < 10 else random.randint(3, 6)
    if random_state is None:
        random_state = numpy.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)

def shear(img):
    rows,cols = img.shape
   
    
    pts1 = np.float32([[5,5],[15,5],[5,25]])
    pt1 = random.randint(1, 3)
    pt2 = random.randint(15, 18)
    pts2 = np.float32([[pt1, 5],
                        [pt2,pt1],
                        [5,pt2]])


    M = cv2.getAffineTransform(pts1,pts2)
    temp = cv2.warpAffine(img,M,(cols,rows))
    return temp


def median_blur(img):
    temp = cv2.medianBlur(img, 1)
    return temp

def blur(img):
    try:
        temp = cv2.blur(img, (random.randint(1, 1), random.randint(1, 1)))
    except:
        temp = img
    return temp    

def add_noise(img):
    random_noise_mask = np.random.choice([0, 255], size = np.shape(img), p = [0.85, 0.15])
    random_noise_mask = np.matrix(random_noise_mask, dtype = np.uint8)
    random_noise_mask = np.bitwise_and(random_noise_mask, img)
    temp = np.bitwise_xor(random_noise_mask, img)
    return temp

def erode(img):
    temp = dilate(img)
    kernel = np.ones((1,2), np.uint8)
    temp = cv2.erode(temp, kernel, iterations=1)
    return temp

def dilate(img):
    kernel = np.ones((2,2), np.uint8)
    temp = cv2.dilate(img, kernel, iterations=1)
    return temp

def loss(img):
    return cv2.resize(cv2.resize(img, (0,0), fx=0.5, fy=0.5), (0, 0), fx = 2, fy = 2)

def resize(img):
    original_rows, original_cols = img.shape
    scalex = 1 +  (random.choice([1, -1]) * (random.random() / 4))
    scaley = 1 +  (random.choice([1, -1]) * (random.random() / 4))
    resized = cv2.resize(img, (0,0), fx=scalex, fy=scaley)
    rows,cols = resized.shape
    temp = np.zeros(np.shape(img), dtype = np.uint8)
    temp[0:min(original_rows, rows), 0:min(original_cols, cols)] = resized[0:min(original_rows, rows), 0:min(original_cols, cols)]
    return temp
    

def opening(img):
    kernel = np.ones((1,1), np.uint8)
    temp = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return temp

def rotate(img):
    angle = random.randint(-15, 15)
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    temp = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
    return temp

def add_random_speckles(img, speckle_num = 2):
    temp = copy.deepcopy(img)
    for i in range(speckle_num):
        randomrow = random.randint(0, 30)
        randomline = random.randint(0, 30)
        cv2.circle(temp, (randomrow, randomline), random.randint(1, 3), (255, 255, 255), -1)
        #img[randomrow : min(30, randomrow + random.randint(0, 8)), randomline : min(30, randomline + random.randint(0, 8)) ] = 0
    return temp

def show(img):
    cv2.imshow('Eorso', img)
    cv2.waitKey(72000)
    cv2.destroyAllWindows()
    
functions = [add_random_speckles,
             rotate,
             opening,
             resize,
             loss,
             add_noise,
             add_random_speckles,
             dilate,
             erode,
             blur,
             median_blur,
             elastic_transform]

overapply = [add_noise,
             elastic_transform,
             resize,
             rotate]

fonts = ['중앙태명조.ttf', 
         '중앙태명조.ttf', 
         'BareunBatangB.ttf', 
         'BareunBatangL.ttf', 
         '직지1950M.TTF',
         '직지강아지M.ttf',
         '-[한글]황진이체.ttf',
         'JejuMyeongjo.ttf', 
         'NanumGothicExtraBold.ttf',
         'SJ아이스베이비.ttf',
         '성동명조B.ttf',
         '자연Regular.ttf',
         'HYCYSM.TTF',
         '자연Bold.ttf',
         '춘풍.ttf',
         'malgun.ttf',
         'HYGSRB.TTF',
         'malgunbd.ttf',
         'HYGPRM.TTF',
         'malgunsl.ttf',
         'KCC-KP-CheongPong-Bold-KP-2011KPS.ttf',
         'KCC-KP-CheongPong-Light-KP-2011KPS.ttf',
         '눈누난나체',
         '-야화L.ttf',
         '윤러브레터체.ttf',
         '백묵달을삼킨연못체(견중).ttf',
         '-불탄고딕B.ttf',
         'KCC-KP-CheongPong-Medium-KP-2011KPS.ttf',
         'KCC-KP-CheonRiMa-Bold-KP-2011KPS.ttf',
         'KCC-KP-CheonRiMa-Light-KP-2011KPS.ttf',
         'KCC-KP-CheonRiMa-Medium-KP-2011KPS.ttf',
         'KCC-KP-CheonRiMa-Normal-KP-2011KPS.ttf',
         'KCC-KP-CR_PyeonChe-Bold-KP-2011KPS.ttf',
         'KCC-KP-CR_Tungkeun-Medium-KP-2011KPS.ttf',
         'KCC-KP-KwangMyeong-Bold-KP-2011KPS.ttf',
         'KCC-KP-KwangMyeong-Medium-KP-2011KPS.ttf',
         'KCC-KP-PK_KungChe-Medium-KP-2011KPS.ttf',
         'KCC-KP-PK_Yeso-Bold-KP-2011KPS.ttf',
         'KCC-KP-PusKul-Medium-KP-2011KPS.ttf',
         'kpcholim.ttf',
         'kpchopom.ttf',
         'kpkwamym.ttf',
         'kppuskum.ttf',
         '조선일보명조.ttf',
         '함초롱바탕R.ttf',
         '210 Soopilmyungjo 020.ttf',
         'BMHANNA_11yrs_ttf.ttf',
         'BMJUA_ttf.ttf',
         'BMKIRANGHAERANG-TTF.ttf',
         'BMYEONSUNG_ttf.ttf',
         'Daum_Regular.ttf',
         'Daum_SemiBold.ttf',
         'DXShnm-KSCpc-EUC-H.ttf',
         'DX명조 10.ttf',
         'Cre사계절M.ttf',
         'DX명조 20.ttf',
         'DX명조 30.ttf',
         'HYSUPB.TTF',
         'DX명조 40.ttf',
         'DX명조 50.ttf',
         'DX명조 60.ttf',
         'EBS훈민정음L.ttf',
         'EBS훈민정음R.ttf',
         'HYMPRL.TTF',
         'HYGTRE.TTF',
         'HYKANB.TTF',
         'HYSANB.TTF',    
         'EBS훈민정음SB.ttf',
         'Pnh복고만화체.ttf',
         'Hankyoreh_Font.TTF',
         'SDMiSaeng.ttf',
         'SeoulHangangB.ttf',
         'SeoulHangangEB.ttf',
         'SeoulHangangL.ttf',
         'SeoulHangangM.ttf',
         'SeoulNamsanB.ttf',
         'SeoulNamsanEB.ttf',
         'SeoulNamsanL.ttf',
         'SeoulNamsanM.ttf',
         'SeoulNamsanvert.ttf',
         '경기천년바탕_Bold.ttf',
         '경기천년바탕_Regular.ttf',
         '경기천년제목V_Bold.ttf',
         '경기천년제목_Bold.ttf',
         '경기천년제목_Light.ttf',
         'MH.TTF',
         'IropkeBatangM.ttf',
         'NanumGothicExtraBold.ttf',
         '성동명조B.ttf',
         'Typo_BallerinaL.ttf',
         'Typo_SSiMyungJo170.ttf',
         '서울한강 장체B.ttf',
         'KCC-KP-CheongPong-Light-KP-2011KPS.ttf',
         'KCC-KP-CheongPong-Bold-KP-2011KPS.ttf',
         'HYWULB.TTF',
         'J빨간구두M.TTF',
         'Navi 몽당연필M.ttf',
         'Navi_모퉁이M.ttf',
         'Navi나른고양이.ttf',
         'Navi상상M.ttf',
         'PNH숲의향.ttf',
         'PNH하늘구름.ttf',
         'rix_다람쥐.ttf',
         'Rix받아쓰기M.ttf',
         'rix삐에로l.ttf',
         'Rix조각공원02.ttf',
         'SJ둥굴레.ttf',
         'SJ보리.ttf',
         'SJ아이스크림.ttf',
         'HYMJRE.TTF',
         'SJ주목나무.ttf',
         'UNI_HSR.TTF',
         '굵은소금.TTF',
         '그남자.TTF',
         '박정아.ttf',
         '사춘기m.ttf',
         '산돌광수명조M.ttf',
         '산돌광수타이프L.ttf',
         '좋은_굿모닝.ttf',
         '좋은_꼭꼭숨어라.ttf',
         '좋은_따뜻한아이_12px.ttf',
         '좋은_롤링페이퍼.ttf',
         '좋은_좋은명조_12px.ttf',
         '좋은꼬마명조체.ttf',
         '좋은꽃사슴.ttf',
         '코스모스d.ttf',
         '쿠키코코.ttf',
         '흑백영화L.ttf']


letters = open('charset.txt', encoding = 'utf-8').read().splitlines()

base_path = os.path.join(path, 'data')

for letter in letters:
    print(letter)
    index = 0
    
    for fontindex, zefont in enumerate(fonts):
        if fontindex % 4 == 3:
            path = os.path.join(base_path, 'test')
        else:
            path = os.path.join(base_path, 'train')
        
        if not os.path.exists(os.path.join(path, str(ord(letter)))):
            os.makedirs(os.path.join(path, str(ord(letter))))
            
        img = Image.new('1', (30, 30), 'white')
        font = ImageFont.truetype(zefont, 25)
        draw = ImageDraw.Draw(img) 
        draw.text((1, 1), letter, font=font) 
        cv = 255 - np.array(img, dtype = np.uint8)
        cv[cv == 254] = 0
        ret2,cv = cv2.threshold(cv,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        for function in functions:
            transformed_pic = function(cv)
            filename = str(index) + '_' + str(function.__name__) + '.jpg'
            fullpath = os.path.join(path, str(ord(letter)), filename)
            fullpath = fullpath.replace('\\', '/')
            cv2.imwrite(fullpath, transformed_pic)
            index += 1
            for new_funct in overapply:
                #new_funct = overapply[random.randint(0, 2)]
                second = new_funct(transformed_pic)
                filename = str(index) + zefont.split('.')[0] + '_' + str(function.__name__) + '_' + str(new_funct.__name__) + '.jpg'
                fullpath = os.path.join(path, str(ord(letter)), filename)
                fullpath = fullpath.replace('\\', '/')
                cv2.imwrite(fullpath, second)
                index += 1