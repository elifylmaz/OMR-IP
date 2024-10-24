import cv2
import numpy as np
import sys
import os

# Modül dizin yolunu ekleyin
sys.path.append(os.path.abspath(".../utlis.py"))

import utlis

# Kameradan görüntü yakalama
cap = cv2.VideoCapture(0)

# Video akışının başlamasını kontrol edin
if not cap.isOpened():
    print("Kamera açılamadı.")
    sys.exit()

widthImg, heightImg = 600, 700  # Örnek boyutlar
questions = 10
choices = 5
ans = [1, 4, 2, 0, 3, 0, 1, 0, 1, 2]

while True:
    ret, img = cap.read()
    if not ret:
        print("Görüntü yakalanamadı.")
        break

    # Fotoğrafı 90 derece saat yönünde döndür
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Dikdörtgen konturları seç
    rectCon = utlis.rectContour(contours)

    # rectCon listesinin içeriğini kontrol et
    if len(rectCon) > 0:
        biggestContour = utlis.getCornerPoints(rectCon[0])

        cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
        biggestContour = utlis.reorder(biggestContour)

        pt1 = np.float32(biggestContour)
        pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # Sağ ve sol kenar konturlarını sayma
        def count_contours_side(image, side):
            side_img = image[:, :image.shape[1] // 2] if side == 'left' else image[:, image.shape[1] // 2:]
            contours, _ = cv2.findContours(side_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            return len(contours)

        left_contours = count_contours_side(imgCanny, 'left')
        right_contours = count_contours_side(imgCanny, 'right')

        print(f"Left Contours: {left_contours}, Right Contours: {right_contours}")

        if right_contours > left_contours:
            imgWarpColored = cv2.rotate(imgWarpColored, cv2.ROTATE_180)
            print("Image rotated by 180 degrees.")

        # Threshold uygula
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray, 110, 255, cv2.THRESH_BINARY_INV)[1]

        boxes = utlis.splitBoxes(imgThresh)

        # Her kutudaki piksel sayılarını al
        myPixelVal = np.zeros((questions, choices), dtype=np.uint16)
        countC = 0
        countR = 0

        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if countC == choices:
                countR += 1
                countC = 0

        print("Pixel Değerleri:\n", myPixelVal)

        # Index değerlerini bul
        myIndex = []

        for x in range(questions):
            arr = myPixelVal[x]

            # Eğer tüm şıklar belirli bir eşik değerden küçükse boş soru olarak işaretle
            if np.all(arr < 2100):
                myIndex.append(-1)
            else:
                # Eşik değerden büyük olan şıkları say
                high_pixels = np.where(arr >= 2800)[0]

                if len(high_pixels) > 1:
                    myIndex.append(-2)  # Birden fazla şık belirli bir eşik değerden büyükse geçersiz soru
                else:
                    # Tek bir şık belirli bir eşik değerden büyükse
                    max_index = np.argmax(arr)
                    if arr[max_index] >= 4200:
                        myIndex.append(max_index)  # Belirli bir yüksek değerden büyükse doğru şıkkın indeksini ekle
                    else:
                        myIndex.append(-2)  # Eşik değerden düşükse geçersiz soru olarak işaretle

        print("İndex Değerleri:\n", myIndex)

        # Notlandırma
        grading = [1 if ans[x] == myIndex[x] else 0 for x in range(questions)]
        score = (sum(grading) / questions) * 100
        print("Grading:\n", grading)
        print(f"Score: {score}")

        # Cevapları göster
        imgResult = imgWarpColored.copy()
        imgResult = utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)

        # Görüntüleri birleştir
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgContours, imgBiggestContours, imgWarpColored, imgResult])
        imgStacked = utlis.stackImages(imageArray, 0.5)

        cv2.imshow("Stacked Image", imgStacked)

    else:
        print("Dikdörtgen kontur bulunamadı.")

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
