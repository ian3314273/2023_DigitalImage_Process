###D1018614 資訊三丙 陳奕安 影像處理_HW3_MagicBand
import numpy as np
import cv2
import sys
import math
sys.setrecursionlimit(7000000) #防止遞迴深度超出python預設的3000，在此設定為300000(可依據圖片長*寬調整)
count = 1
img = cv2.imread("iphone15_model.jpg")
out_RGB = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8) 
out_HSV = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8) 
out_LAB = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8) 
def RGB2HSV(B, G, R):
    B /= 255
    G /= 255
    R /= 255
    H, S, V = 0, 0, 0
    Max = max(B, G, R)
    Min = min(B, G, R)
    Delta = Max-Min
    V = Max
    if V == 0:
        S = 0
    else:
        S = V - Min
    if V == R:
        H = 60*((G-B)/Delta+0)
    elif V == G:
        H = 60*((B-R)/Delta+2)
    elif V == B:
        H = 60*((R-G)/Delta+4)
    if H < 0:
        H += 360
    H /= 2
    S *= 255
    V *= 255
    return [H, S ,V]
def RGB2LAB(B, G, R):
    Xn = 0.950456
    Yn = 1.0
    Zn = 1.088754
    X = 0.412453 * R + 0.357580 * G + 0.180423 * B
    Y = 0.212671 * R + 0.715160 * G + 0.072169 * B
    Z = 0.019334 * R + 0.119193 * G + 0.950227 * B
    X /= 255 * Xn
    Y /= 255 * Yn
    Z /= 255 * Zn
    if Y > 0.008856:
        fy = pow(Y, (1.0/3.0))
        L = 116.0 * fy - 16.0
    else:
        fy = 7.787 * Y - (16.0/116.0)
        L = 903.3 * fy
    if L < 0:
        L = 0.0
    if X > 0.008856:
        fx = pow(X, (1.0/3.0))
    else:
        fx = 7.787 * X + (16.0/116.0)
    if Z > 0.008856:
        fz = pow(Z, (1.0/3.0))
    else:
        fz = 7.787 * Z + (16.0/116.0)
    A = 500.0*(fx-fy)
    B = 200.0*(fy-fz)
    L *= 255/100
    A += 128
    B += 128
    return [L, A, B]
def Magic_Band(Start_y, Start_x, Threshold):
    visited = np.zeros((img.shape[0], img.shape[1])) #建立全為0之二為矩陣紀錄是否走訪
    visited_HSV = np.zeros((img.shape[0], img.shape[1])) #建立全為0之二為矩陣紀錄是否走訪
    visited_LAB = np.zeros((img.shape[0], img.shape[1]))
    start_RGB = img[Start_x, Start_y] #取得起始點的RGB訊息
    #Deep_First_Search之走訪整個圖片(已為走訪及是否為同一個區塊作為基準)
    def DFS_img(x, y):
        if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1] or visited[x,y] == 1:
            return
        visited[x,y] = 1
        Current_Color = img[x, y]
        if(ThresHold_Check_RGB(Current_Color, start_RGB, Threshold)):
            out_RGB[x, y] = img[x, y]
        else :
            return
        #DFS using 四連通
        DFS_img(x+1, y)
        DFS_img(x-1, y)
        DFS_img(x, y+1)
        DFS_img(x, y-1)
    def DFS_img_HSV(x,y):
        if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1] or visited_HSV[x,y] == 1:
            return
        visited_HSV[x,y] = 1
        Current_Color = img[x, y]
        if(ThresHold_Check_HSV(Current_Color, start_RGB, Threshold)):
            out_HSV[x, y] = img[x, y]
        else :
            return
        #DFS using 四連通
        DFS_img_HSV(x+1, y)
        DFS_img_HSV(x-1, y)
        DFS_img_HSV(x, y+1)
        DFS_img_HSV(x, y-1)
    def DFS_img_LAB(x,y):
        if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1] or visited_LAB[x,y] == 1:
            return
        visited_LAB[x,y] = 1
        Current_Color = img[x, y]
        if(ThresHold_Check_LAB(Current_Color, start_RGB, Threshold)):
            out_LAB[x, y] = img[x, y]
        else :
            return
        #DFS using 四連通
        DFS_img_LAB(x+1, y)
        DFS_img_LAB(x-1, y)
        DFS_img_LAB(x, y+1)
        DFS_img_LAB(x, y-1)
    #檢查是否為門檻值內
    def ThresHold_Check_RGB(Current, Target, Threshold):
        dist_Deno = ((Current[0]**2 + Current[1]**2 + Current[2]**2)**0.5)*((Target[0]**2 + Target[1]**2 + Target[2]**2)**0.5)
        dist_Nume = (Target[0]*1) * (Current[0]*1) + (Target[1]*1) * (Current[1]*1) + (Target[2]*1) * (Current[2]*1)
        if(dist_Nume > dist_Deno):
            dist_Nume -= 0.1
        dist = dist_Nume / dist_Deno
        if (math.acos(dist)*180/math.pi) <= Threshold:
            return True
        else:
            return False
    
    #Transform RGM domain to HSV and check
    def ThresHold_Check_HSV(Current, Target, Threshold):
        Current = RGB2HSV(Current[0]*1, Current[1]*1, Current[2]*1)
        Target = RGB2HSV(Target[0]*1, Target[1]*1, Target[2]*1)
        dist_Deno = ((Current[0]**2 + Current[1]**2 + Current[2]**2)**0.5)*((Target[0]**2 + Target[1]**2 + Target[2]**2)**0.5)
        dist_Nume = (Target[0]*1) * (Current[0]*1) + (Target[1]*1) * (Current[1]*1) + (Target[2]*1) * (Current[2]*1)
        if(dist_Nume > dist_Deno):
            dist_Nume -= 0.1
        dist = dist_Nume / dist_Deno
        if (math.acos(dist)*180/math.pi) <= Threshold:
            return True
        else:
            return False
    def ThresHold_Check_LAB(Current, Target, Threshold):
        Current = RGB2LAB(Current[0]*1, Current[1]*1, Current[2]*1)
        Target = RGB2LAB(Target[0]*1, Target[1]*1, Target[2]*1)
        dist_Deno = ((Current[0]**2 + Current[1]**2 + Current[2]**2)**0.5)*((Target[0]**2 + Target[1]**2 + Target[2]**2)**0.5)
        dist_Nume = (Target[0]*1) * (Current[0]*1) + (Target[1]*1) * (Current[1]*1) + (Target[2]*1) * (Current[2]*1)
        if(dist_Nume > dist_Deno):
            dist_Nume -= 0.1
        dist = dist_Nume / dist_Deno
        if (math.acos(dist)*180/math.pi) <= Threshold:
            return True
        else:
            return False
    #用RGB和HSV各跑一次    
    DFS_img(Start_x, Start_y)
    DFS_img_HSV(Start_x, Start_y)
    DFS_img_LAB(Start_x, Start_y)
def main():
    Threshold, Start_y, Start_x = input().split()
    Magic_Band(int(Start_y), int(Start_x), int(Threshold))
    cv2.imshow("Original", img)
    cv2.imshow("MagicBand_Base_RGB", out_RGB)
    cv2.imshow("MagicBand_Base_HSV", out_HSV)
    cv2.imshow("MagicBand_Base_LAB", out_LAB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
