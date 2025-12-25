# OCR
#  from PIL import Image
# import cv2
# import pytesseract
#
# img = cv2.imread("Schermata del 2025-12-14 16-02-55.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#
# text = pytesseract.image_to_string(thresh, config="--psm 6")
# print(text)


import pdfplumber

with pdfplumber.open("Alessandro_Fella_CV.pdf") as pdf:
    first_page = pdf.pages[0]
    print(first_page.extract_text())
