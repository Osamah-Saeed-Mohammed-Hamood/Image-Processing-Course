import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# اختيار الصورة
# ---------------------------
img_path = r"D:\IT\Level 4\Image Processing\Image-Processing-Course\Lab5\HW edge detection\images\test2.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# ---------------------------
# اختيار المشغل
# ---------------------------
# خيارات المشغل: "Sobel", "Prewitt", "Roberts", "Gradient", "Laplacian", "Kirsch", "Robinson", "LoG"
operator = "Sobel"  # غير هذا لاختبار مشغل مختلف

# ---------------------------
# Sobel Operator
# ---------------------------
if operator == "Sobel":
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    combined = cv2.magnitude(sobel_x, sobel_y)
    plt.subplot(1,3,1), plt.imshow(sobel_x, cmap='gray'), plt.title("Sobel X")
    plt.subplot(1,3,2), plt.imshow(sobel_y, cmap='gray'), plt.title("Sobel Y")
    plt.subplot(1,3,3), plt.imshow(combined, cmap='gray'), plt.title("Sobel Combined")

# ---------------------------
# Prewitt Operator
# ---------------------------
elif operator == "Prewitt":
    prewitt_x = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
    prewitt_y = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])
    px = cv2.filter2D(img, -1, prewitt_x)
    py = cv2.filter2D(img, -1, prewitt_y)
    combined = cv2.magnitude(np.float32(px), np.float32(py))
    plt.subplot(1,3,1), plt.imshow(px, cmap='gray'), plt.title("Prewitt X")
    plt.subplot(1,3,2), plt.imshow(py, cmap='gray'), plt.title("Prewitt Y")
    plt.subplot(1,3,3), plt.imshow(combined, cmap='gray'), plt.title("Prewitt Combined")

# ---------------------------
# Roberts Operator
# ---------------------------
elif operator == "Roberts":
    ro_x = np.array([[1,0],[0,-1]], dtype=np.float32)
    ro_y = np.array([[0,1],[-1,0]], dtype=np.float32)
    rx = cv2.filter2D(img, -1, ro_x)
    ry = cv2.filter2D(img, -1, ro_y)
    combined = cv2.magnitude(np.float32(rx), np.float32(ry))
    plt.subplot(1,3,1), plt.imshow(rx, cmap='gray'), plt.title("Roberts X")
    plt.subplot(1,3,2), plt.imshow(ry, cmap='gray'), plt.title("Roberts Y")
    plt.subplot(1,3,3), plt.imshow(combined, cmap='gray'), plt.title("Roberts Combined")

# ---------------------------
# Gradient Operator
# ---------------------------
elif operator == "Gradient":
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    plt.subplot(1,3,1), plt.imshow(grad_x, cmap='gray'), plt.title("Gradient X")
    plt.subplot(1,3,2), plt.imshow(grad_y, cmap='gray'), plt.title("Gradient Y")
    plt.subplot(1,3,3), plt.imshow(magnitude, cmap='gray'), plt.title("Gradient Magnitude")

# ---------------------------
# Laplacian Operator
# ---------------------------
elif operator == "Laplacian":
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title("Original")
    plt.subplot(1,2,2), plt.imshow(laplacian, cmap='gray'), plt.title("Laplacian")

# ---------------------------
# Kirsch Compass Masks
# ---------------------------
elif operator == "Kirsch":
    kirsch_masks = {
        "N": np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]],dtype=np.float32),
        "NE": np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]],dtype=np.float32),
        "E": np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]],dtype=np.float32),
        "SE": np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]],dtype=np.float32),
        "S": np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]],dtype=np.float32),
        "SW": np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]],dtype=np.float32),
        "W": np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]],dtype=np.float32),
        "NW": np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]],dtype=np.float32)
    }
    plt.figure(figsize=(12,8))
    plt.subplot(3,3,1), plt.imshow(img,cmap='gray'), plt.title("Original")
    i=2
    for d,k in kirsch_masks.items():
        plt.subplot(3,3,i), plt.imshow(cv2.filter2D(img,-1,k), cmap='gray'), plt.title(d)
        i+=1
    plt.tight_layout()
    plt.show()

# ---------------------------
# Robinson Compass Masks
# ---------------------------
elif operator == "Robinson":
    robinson_masks = {
        "N": np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32),
        "NE": np.array([[0,1,2],[-1,0,1],[-2,-1,0]],dtype=np.float32),
        "E": np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float32),
        "SE": np.array([[2,1,0],[1,0,-1],[0,-1,-2]],dtype=np.float32),
        "S": np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float32),
        "SW": np.array([[0,-1,-2],[1,0,-1],[2,1,0]],dtype=np.float32),
        "W": np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float32),
        "NW": np.array([[-2,-1,0],[-1,0,1],[0,1,2]],dtype=np.float32)
    }
    plt.figure(figsize=(12,8))
    plt.subplot(3,3,1), plt.imshow(img,cmap='gray'), plt.title("Original")
    i=2
    for d,k in robinson_masks.items():
        plt.subplot(3,3,i), plt.imshow(cv2.filter2D(img,-1,k), cmap='gray'), plt.title(d)
        i+=1
    plt.tight_layout()
    plt.show()

# ---------------------------
# Laplacian of Gaussian (LoG)
# ---------------------------
elif operator == "LoG":
    log_kernel = np.array([[0,0,-1,0,0],
                            [0,-1,-2,-1,0],
                            [-1,-2,16,-2,-1],
                            [0,-1,-2,-1,0],
                            [0,0,-1,0,0]], dtype=np.float32)
    log_result = cv2.filter2D(img,-1,log_kernel)
    plt.subplot(1,2,1), plt.imshow(img,cmap='gray'), plt.title("Original")
    plt.subplot(1,2,2), plt.imshow(log_result,cmap='gray'), plt.title("LoG 5x5")

plt.show()
