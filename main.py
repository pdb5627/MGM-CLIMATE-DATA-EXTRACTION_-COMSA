## Author: ERDEM CANAZ
## email : erdemcanaz@gmail.com
## Modifications by Paul Brown
## https://github.com/user/pdb5627
import string

from PIL import Image
import numpy as np
import pytesseract
import os, sys
import csv
import datetime
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageParseError(Exception):
    """ Raised when there is a problem getting data out of the meteogram image. """
    pass


horizontal_pixels = [79,89,99,110,120,131,141,152,162,172,183,193,204,214,225,235,245,256,266,277,287,298,308,319,
                     329,339,350,360,371,381,392,402,412,423,433,444,454,465,475,486,496,506,517,527,538,548,559,569,
                     579,590,600,611,621,632,642,653,663,673,684,694,705,715,726,736,746,757,767,778,788,799,809,820]
top_and_bottoms = [119,234,313,528,589,705] # cloud, temperature, rain
mm_per_mSquare_per_px = 0.0650 # if there is no catastrophic event, good approximation for ankara model!


def extract_data(image_array, which_data):
    #which_data-> 0: cloud 1:temp 2:rain
    return_data = []

    for x in horizontal_pixels: ###SCAN COLUMN-S
        is_zero = True

        for y in range (top_and_bottoms[which_data*2], top_and_bottoms[which_data*2 +1]): ###SCAN COLUMN

            R = int(image_array[y][x][0])
            G = int(image_array[y][x][1])
            B = int(image_array[y][x][2])
            is_gray = abs(R - G) <= 10 and abs(G - B) <= 10
            #this condition not only considers white but olso all binary colors such as scale lines, time lines, anything black
            if not is_gray:
                if which_data == 0: #cloud
                    percentage_dec = (top_and_bottoms[1]-y) / (top_and_bottoms[1] - top_and_bottoms[0])
                    return_data.append(percentage_dec)
                    break
                elif which_data == 2: #rain
                    rain_mm = (top_and_bottoms[5] - y) * mm_per_mSquare_per_px
                    return_data.append(rain_mm)
                    break
                elif which_data == 1: # temperature
                    # Look for just red line or red and green overlapping
                    if R - G > 30 or (G > 160 and R / B > 1.5):
                        return_data.append(y)
                        break
        ####IF NO DATA IS FOUND
        else:
            if which_data == 1: #temperature
                if len(return_data) > 0:
                    logger.warning(f'Point on temperature line not detected. Filling with previous value. x = {x} px')
                    return_data.append(return_data[-1])
                else:
                    logger.warning(f'Point on temperature line not detected. Filling with NAN. x = {x} px')
                    return_data.append(np.nan)
            else: #cloud & rain
                return_data.append(0)

    return return_data


def scale_temperature_pixels (image_array, image, temperature_pixel_data_array):
    temp_graph_left = 80
    temp_scale_top = 296
    temp_scale_bottom = 528
    vertical_candidates = []
    values_of_temp_scale_lines = []
    for y in range(temp_scale_top, temp_scale_bottom):
        R = int(image_array[y][temp_graph_left][0])
        G = int(image_array[y][temp_graph_left][1])
        B = int(image_array[y][temp_graph_left][2])
        is_gray = abs(R - G) <= 10 and abs(G - B) <= 10
        if is_gray and R < 250:
            line_y_coordinate = y
            temp_scale = image.crop((40, line_y_coordinate - 10, 74, line_y_coordinate + 10))
            # PSM modes: https://github.com/tesseract-ocr/tesseract/issues/434
            temp_scale_text = pytesseract.image_to_string(temp_scale, config='--psm 7')
            tbl = str.maketrans('O', '0', ',%!$' + string.ascii_lowercase + string.ascii_uppercase)
            temp_scale_text = temp_scale_text.strip().translate(tbl)
            try:
                values_of_temp_scale_lines.append(float(temp_scale_text))
            except ValueError:
                continue
            vertical_candidates.append(y)

    # Sanity check on values. Values go top to bottom, so they should be decreasing.
    pts = (0, len(values_of_temp_scale_lines)-1)  # Default to first and last point
    if values_of_temp_scale_lines[0] <= values_of_temp_scale_lines[1] \
            and len(values_of_temp_scale_lines) > 2 \
            and values_of_temp_scale_lines[1] > values_of_temp_scale_lines[2]:
        # Point 0 is bogus, use Points 1 and 2
        pts = (1, 2)
    real_temperatures = []
    dt = values_of_temp_scale_lines[pts[0]] - values_of_temp_scale_lines[pts[1]]
    dpx = vertical_candidates[pts[0]] - vertical_candidates[pts[1]]
    slope = dt / dpx
    for px in temperature_pixel_data_array:
        val = (px - vertical_candidates[pts[0]]) * slope + values_of_temp_scale_lines[pts[0]]  # line equation
        real_temperatures.append(val)
    return real_temperatures


def extract_title_region_date(image):
    title = image.crop((338, 16, 560, 41))
    title_text = pytesseract.image_to_string(title, lang='eng').strip()

    region = image.crop((84, 32, 149, 55))
    region_text = pytesseract.image_to_string(region, lang='eng').strip().replace('0', 'O')

    date = image.crop((694, 64, 828, 84))
    date_text = pytesseract.image_to_string(date, lang='eng').strip().replace('O', '0')
    return title_text, region_text, date_text


def extract_meteogram_data(image):
    image_array = np.array(image)

    title_text, region_text, date_text = extract_title_region_date(image)

    if(title_text != "WRF METEOGRAM"):
       raise ImageParseError("Image does not appear to be as expected. Text 'WRF METEOGRAM' not found.")

    current_dt = pd.to_datetime(date_text, format='%d/%m/%Y %H%Z')
    dt = pd.date_range(current_dt, periods=72, freq='1H')

    cloud = extract_data(image_array, 0)

    temperature_pixels=extract_data(image_array, 1)
    temperature = scale_temperature_pixels(image_array, image, temperature_pixels)

    rain = extract_data(image_array, 2)

    df = pd.DataFrame({'location': region_text,
                       'current_dt': current_dt,
                       'dt': dt,
                       'cloud': cloud,
                       'temperature': temperature,
                       'rain': rain}, index=dt)

    return df


def main(argv=None):
    if argv is None:
        argv = sys.argv

    for fname in argv[1:]:
        file, ext = os.path.splitext(fname)
        logger.info("Processing file:" + fname)
        image = Image.open(fname)
        df = extract_meteogram_data(image)
        df.to_csv(file+'.csv', index=False)


if __name__ == '__main__':
    sys.exit(main())


























