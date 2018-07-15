import cv2
import copy
import numpy as np
from sklearn.neighbors import KDTree


def show(img):
    cv2.imshow('Eorso', img)
    cv2.waitKey(72000)
    cv2.destroyAllWindows()

def find_outliers(data, m = 2):
    absdev = abs(data - np.median(data))
    mad = np.median(absdev)
    scaled_dist = absdev / mad if mad else 0
    return scaled_dist > m
    
def minimum_bounding_rectangle(rects):
    list_top = [rect[1] for rect in rects]
    list_left = [rect[0] for rect in rects]
    list_right = [rect[0] + rect[2] for rect in rects]
    list_bottom = [rect[1] + rect[3] for rect in rects]
    min_top = min(list_top)
    min_left = min(list_left)
    max_right = max(list_right)
    max_bottom = max(list_bottom)
    new_rect = (min_left,
                min_top,
                max_right - min_left,
                max_bottom - min_top)
    return new_rect

img = cv2.imread('F:/OCR/sample.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
im2, contours, hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_cont_area = round((np.shape(img)[0] * np.shape(img)[0]) * .2)


list_area = []
list_rect = []
new_image = np.zeros(np.shape(th2), dtype=np.uint8)
new_image.fill(255)

for contour in contours:
    x, y, width, height = cv2.boundingRect(contour)
    area = width * height
    if area > 50 and area < max_cont_area:
        list_rect.append((x, y, width, height))
        list_area.append(width * height)
        new_image[y : y + height, x: x + width] = th2[y : y + height, x: x + width]
        
kernel = np.ones((8,2), np.uint8)
img_transform = cv2.morphologyEx(new_image, cv2.MORPH_OPEN, kernel)
img_transform = 255 - img_transform
im2, contours, hierarchy = cv2.findContours(img_transform,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

list_area = []
list_rect = []
for contour in contours:
    x, y, width, height = cv2.boundingRect(contour)
    area = width * height
    if area > 50:
        list_area.append(area)
        list_rect.append((x, y, width, height))



outliers = find_outliers(list_area, 3)
median = np.median(list_area)
elements_to_merge = []
list_x = []
list_y = []
list_heights = []
list_widths = []

for i, rect in enumerate(list_rect):
    x, y, width, height = rect
    list_x.append(x)
    list_y.append(y)
    list_widths.append(width)
    list_heights.append(height)
    if outliers[i] == True and width * height < median:
        elements_to_merge.append(i)  
     
    
median_width = np.median(list_widths)
median_height = np.median(list_heights)
centers_x = np.add(list_x, list_widths) / 2
centers_y = np.add(list_y, list_heights) / 2
coordinates = list(zip(centers_x, centers_y))
tree = KDTree(coordinates)

merged_elements = []
clean_list = list(set(list_rect))

for element in elements_to_merge:
    dist, ind = tree.query([coordinates[element]], 2)
    mbr = minimum_bounding_rectangle([list_rect[element], list_rect[ind[0][1]]])
    try:
        clean_list.pop(clean_list.index(list_rect[element]))
    except:
        pass
    if ((mbr[2] < median_width * 1.5) and
        (mbr[3] < median_height * 1.3)):
        try:
            clean_list.pop(clean_list.index(list_rect[ind[0][1]]))
        except:
            pass
        merged_elements.append(mbr)
        clean_list.append(mbr)


clean_list = list(set(clean_list))

coordinates = [((l[0] + l[2]) / 2, (l[1] + l[3])/  2) for l in clean_list]
tree = KDTree(coordinates)

final_list = list(clean_list)
for element in merged_elements:
    dist, ind = tree.query([((element[0] + element[2]) / 2, (element[1] + element[3]) / 2)], 2)
    nearest_neighbor = ind[0][1]

    if (element[0] >= clean_list[nearest_neighbor][0] and
        element[1] >= clean_list[nearest_neighbor][1] and
        element[0] + element[2] <= clean_list[nearest_neighbor][0] + clean_list[nearest_neighbor][2] and
        element[1] + element[3] <= clean_list[nearest_neighbor][1] + clean_list[nearest_neighbor][3] and
        clean_list[nearest_neighbor] != element):
        try:
            final_list.pop(final_list.index(element))
        except:
            pass


clean_image = np.zeros(np.shape(th2), dtype=np.uint8)
clean_image.fill(255)
     
for rect in final_list:
    clean_image[rect[1] : rect[1] + rect[3], rect[0]: rect[0] + rect[2]] = new_image[rect[1] : rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    

img2 = cv2.cvtColor(copy.deepcopy(clean_image), cv2.COLOR_GRAY2BGR)
for i, rect in enumerate(final_list):
    cv2.rectangle(img2,(rect[0],rect[1]),(rect[0] + rect[2], rect[1] + rect[3]),(0,0,250),2)
    cut = clean_image[rect[1] : rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    cut = 255 - cut
    resized = cv2.resize(cut, (30, 30), interpolation = cv2.INTER_AREA)
    cv2.imwrite('data/' + str(i) + '.jpg', resized)
    #cv2.putText(img2, str(i), (rect[0],rect[1] - random.randint(10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 155, 55))
