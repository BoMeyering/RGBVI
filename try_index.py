from rgbvi import compute_index, exg, index_specs
# from rgbvi.indices.linear import exg
import numpy as np
import cv2
import matplotlib.cm as cm

from glob import glob

def show_img(title, image):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

images = sorted(glob("*.jpg", root_dir="./data/images"))
labels = sorted(glob("*.png", root_dir="./data/labels"))

for img_name, mask_name in zip(images, labels):
    print("Image:", img_name, "Mask:", mask_name)
    img = cv2.imread("./data/images/" + img_name, cv2.IMREAD_COLOR_RGB)
    show_img("Original", img)
    label = cv2.imread("./data/labels/" + mask_name, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_GRAYSCALE)
    mask = np.where(np.isin(label, [2, 3, 4, 5]), 1, 0).astype(np.uint8)
    # show_img("Mask Orig", mask*255)
    antimask = mask ^ 1
    antiimg = cv2.bitwise_and(img, img, mask=antimask)
    # show_img("antiMask", antimask*255)
    # show_img("Mask", antiimg)

    indx = compute_index("vari", img, robust_mean=True)
    # indx = (indx - np.min(indx)) / (np.max(indx) - np.min(indx))q
    print(indx.min(), indx.max())
    print("Index min/max:", np.min(indx), np.max(indx))
    
    # Convert to 8-bit image
    indx_8bit = (indx*255).astype(np.uint8)
    show_img("Index Grayscale", indx_8bit)
    
    # Apply color palette (colormap)
    # Options: cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA, 
    #          cv2.COLORMAP_INFERNO, cv2.COLORMAP_TURBO, cv2.COLORMAP_HOT, etc.
    colored_indx = cv2.applyColorMap(indx_8bit, cv2.COLORMAP_PLASMA)
    colored_indx = cv2.cvtColor(colored_indx, cv2.COLOR_BGR2RGB)
    # show_img("Index Color", colored_indx)
    
    # Optional: Apply mask to show only vegetation
    # if mask is not None:
    #     colored_indx[mask == 0] = 0  # Set non-vegetation pixels to black
    
    # show_img("Index", colored_indx)
    combo = cv2.bitwise_or(antiimg, colored_indx)
    # show_img("Combo", combo)
    show_img('index', colored_indx)

    # mask = (mask > 0).astype(np.bool_)
    # indx = compute_index("rgbvi", img, mask=mask, robust_mean=False)
    # print(indx.shape)
    # print("IDX:\n", indx)

    # print("Min:", np.min(indx), "Max:", np.max(indx), "Mean:", np.mean(indx))
    # indx = (indx - np.min(indx)) / (np.max(indx) - np.min(indx))
    # indx = (indx*255).astype(np.uint8)      
    # cv2.imwrite(f"./data/indices/{img_name[:-4]}_rgbvi.png", indx)
cv2.destroyAllWindows()



# img = cv2.imread("Ta00034_20170703.jpg", cv2.IMREAD_COLOR_RGB)
# # img_2 = img / 255.0
# # img_2 = np.where(img_2==0, 1e-3, img_2)
# # print(img_2.min(), img_2.max())
# # R, G, B = img_2[..., 0], img_2[..., 1], img_2[..., 2]
# # indx = 0.5268 * ((R ** -0.1294) * (G ** 0.3389) * (B ** -0.3118))

# indx = compute_index("rgbvi", img, robust_mean=False)
# print(indx.shape)
# print("IDX:\n", indx)

# print("Min:", np.min(indx), "Max:", np.max(indx), "Mean:", np.mean(indx))
# indx = (indx - np.min(indx)) / (np.max(indx) - np.min(indx))
# indx = (indx*255).astype(np.uint8)      
# show_img("index", indx)

# # show_img("original", indx)

# mask = (indx >=140).astype(np.bool_)
# masked = img.copy()
# masked[~mask] = 0
# show_img("masked", masked)