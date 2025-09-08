import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# الدالة اليدوية للـ Mean Filter
# تستقبل صورة img وحجم نافذة ksize (الافتراضي 5).
def mean_filter_manual(img, ksize=5):
    # نصف حجم النافذة (مثلاً 5 // 2 = 2) لاستخدامه في الحواف.
    pad = ksize // 2
    # نضيف إطار للصورة بمقدار pad من كل جهة باستخدام انعكاس الحواف BORDER_REFLECT حتى نقدر نأخذ نوافذ كاملة عند الأطراف بدون فقد.
    padded_img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REFLECT)

    filtered_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # نأخذ النافذة الحالية
            window = padded_img[i:i+ksize, j:j+ksize]
            # حساب المتوسط المكاني على بعديّ الصف/العمود فقط (0 و1)،
            filtered_img[i, j] = np.mean(window, axis=(0,1))
    # نحول نوع البيانات للصورة المفلترة إلى uint8
    return filtered_img.astype(np.uint8)


cap = cv.VideoCapture('videos/video.mp4')  # أو 0 للكاميرا

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # تطبيق فلتر المتوسط يدويًا على الإطار
    mean_manual = mean_filter_manual(frame, ksize=5)
    # فلتر جاهز
    mean_ready = cv.blur(frame, (5,5))

    # عرض الفيديو
    cv.imshow('Original Video', frame)
    cv.imshow('Mean Filter (Manual)', mean_manual)
    cv.imshow('Mean Filter (cv.blur)', mean_ready)
    
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# اغلاق ملف/جهاز الفيديو وتحرير الموارد
cap.release()
cv.destroyAllWindows()
