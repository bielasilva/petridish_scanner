import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


from PIL import ImageFont, ImageDraw, Image

def cm2inch(value):
    inch = 2.54
    return value/inch

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    plt.imsave(buf, fig, cmap='gray')
    buf.seek(0)
    img = Image.open(buf)
    return img


# Set up parameters for reference sheet 
w_in, h_in = (11, 8.5) # Real label size in inches
res = 1500              # Desired resolution DPI
dpcm = res / 2.54      # Dots per cm

# Calculate the size of the image in pixels
w = int(res * w_in)
h = int(res * h_in)



aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
mkr_size = [0, int( res * cm2inch(1) ), int( res * cm2inch(2) )] # 1 inch = 2.54 cm


aruco1_ids = [1,2,3,4,0,0,0,0]
aruco2_ids = [5,6,7,8,9,10,11,12]

aruco_mkrs = {"aruco1":[], "aruco2":[]}

for i in range(len(aruco2_ids)):
    aruco1 = cv2.aruco.generateImageMarker(aruco_dict, aruco1_ids[i], mkr_size[1])
    aruco2 = cv2.aruco.generateImageMarker(aruco_dict, aruco2_ids[i], mkr_size[2])
    
    aruco_mkrs["aruco1"].append(aruco1)
    aruco_mkrs["aruco2"].append(aruco2)


# Create the reference sheet
for size_fact in [0, 1, 2, 3]:
    ref_sheet = Image.new('RGB', size = (w, h), color=(255,255,255))
    
    # Left upper corner
    ref_sheet.paste(fig2img(aruco_mkrs['aruco2'][0]), (int(dpcm * (1 + size_fact)), int(dpcm * (1 + size_fact))))
    ref_sheet.paste(fig2img(aruco_mkrs['aruco1'][0]), (int(dpcm * (3 + size_fact)), int(dpcm * (3 + size_fact))))

    # Left bottom corner
    ref_sheet.paste(fig2img(aruco_mkrs['aruco1'][1]), (int(dpcm * (3 + size_fact)), h - int(dpcm * (4 + size_fact))))
    ref_sheet.paste(fig2img(aruco_mkrs['aruco2'][1]), (int(dpcm * (1 + size_fact)), h - int(dpcm * (3 + size_fact))))

    # Right upper corner
    ref_sheet.paste(fig2img(aruco_mkrs['aruco2'][2]), (w - int(dpcm * (3 + size_fact)), int(dpcm * (1 + size_fact))))
    ref_sheet.paste(fig2img(aruco_mkrs['aruco1'][2]), (w - int(dpcm * (4 + size_fact)), int(dpcm * (3 + size_fact))))

    # Right bottom corner
    ref_sheet.paste(fig2img(aruco_mkrs['aruco1'][3]), (w - int(dpcm * (4 + size_fact)), h - int(dpcm * (4 + size_fact))))
    ref_sheet.paste(fig2img(aruco_mkrs['aruco2'][3]), (w - int(dpcm * (3 + size_fact)), h - int(dpcm * (3 + size_fact))))

    # Top middle 
    ref_sheet.paste(fig2img(aruco_mkrs['aruco2'][4]), (int(w/2 - dpcm), int(dpcm * (1 + size_fact))))

    # Bottom middle 
    ref_sheet.paste(fig2img(aruco_mkrs['aruco2'][5]), (int(w/2 - dpcm), (h - int(dpcm * (3 + size_fact)))))

    # Left middle 
    ref_sheet.paste(fig2img(aruco_mkrs['aruco2'][6]), (int(dpcm * (1 + size_fact)), int(h/2 - dpcm)))

    # Right middle 
    ref_sheet.paste(fig2img(aruco_mkrs['aruco2'][7]), (w - int(dpcm * (3 + size_fact)), int(h/2 - dpcm)))

    # legend = f"Size Factor: {size_fact} | ArUco IDs LUC: {aruco1_ids[0]}, {aruco2_ids[0]}, LBC: {aruco1_ids[1]}, {aruco2_ids[1]}, RUC: {aruco1_ids[2]}, {aruco2_ids[2]}, RBC: {aruco1_ids[3]}, {aruco2_ids[3]}, TM: {aruco2_ids[4]}, BM: {aruco2_ids[5]}, LM: {aruco2_ids[6]}, RM: {aruco2_ids[7]}"
    legend = f"Size Factor: {size_fact} | IDs [{aruco1_ids[0]}, {aruco2_ids[0]}], {aruco2_ids[4]}, [{aruco1_ids[2]}, {aruco2_ids[2]}], {aruco2_ids[7]}, [{aruco1_ids[3]}, {aruco2_ids[3]}], {aruco2_ids[5]}, [{aruco1_ids[1]}, {aruco2_ids[1]}], {aruco2_ids[6]}"

    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 300)
    draw = ImageDraw.Draw(ref_sheet)
    draw.text((int(dpcm * (1 + size_fact)), int(dpcm * (1 + size_fact) - 500)),legend,(0,0,0), font=font)
    ref_sheet = draw._image

    plt.imshow(ref_sheet, cmap='gray')
    plt.axis('off')

    for fmt in ["png", "pdf"]:
        plt.savefig(f"Reference_sheet_SF{size_fact}.{fmt}", dpi = res, transparent=True, orientation = 'portrait', bbox_inches = 'tight', pad_inches = 0)