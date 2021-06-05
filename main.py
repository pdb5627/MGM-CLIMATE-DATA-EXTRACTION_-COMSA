## Author: ERDEM CANAZ
## email : erdemcanaz@gmail.com
## Modifications by Paul Brown
## https://github.com/user/pdb5627
import string

from PIL import Image
import urllib.request
import numpy as np
import pytesseract
import os, sys
import csv
import datetime

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageParseError(Exception):
    """ Raised when there is a problem getting data out of the meteogram image. """
    pass


horizontal_pixels = [79,89,99,110,120,131,141,152,162,172,183,193,204,214,225,235,245,256,266,277,287,298,308,319,329,339,350,360,371,381,392,402,412,423,433,444,454,465,475,486,496,506,517,527,538,548,559,569,579,590,600,611,621,632,642,653,663,673,684,694,705,715,726,736,746,757,767,778,788,799,809,820]
top_and_bottoms = [119,234,313,528,589,705] # cloud, temperature, rain
decimal_accuracy = [3, 1, 2] # rounds cloud,temperature and rain values
mm_per_mSquare_per_px =0.0650 #if there is no catastrophic event, good approximation for ankara model!


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
                    percentage_dec = round(percentage_dec, decimal_accuracy[0])
                    return_data.append(percentage_dec)
                    break
                elif which_data == 2: #rain
                    rain_mm = (top_and_bottoms[5] - y) * mm_per_mSquare_per_px
                    rain_mm = round(rain_mm, decimal_accuracy[2])
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


def initiliaze_temperature_pixels (image_array, image, temperature_pixel_data_array):
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
        val = round(val, decimal_accuracy[1])
        real_temperatures.append(val)
    return real_temperatures


def get_date_as_number(date_text):
    # Sometimes a 0 might be misread as an O. There is no reason for an O to appear in this text, so change any Os to 0s
    date_text = date_text.replace('O', '0')
    day = 10 * int(date_text[0]) + int(date_text[1])
    month = 10 * int(date_text[3]) + int(date_text[4])
    year = 1000 * int(date_text[6]) + 100*int(date_text[7]) + 10*int(date_text[8]) + int(date_text[9])
    shift = 10 * int(date_text[11]) + int(date_text[12])
    date_array = [year, month, day, shift]
    return date_array


def extract_title_region_date(image):
    title = image.crop((338, 16, 560, 41))
    title_text = pytesseract.image_to_string(title, lang='eng')[:-2]

    region = image.crop((84, 32, 149, 55))
    region_text = pytesseract.image_to_string(region, lang='eng')[:-2]

    date = image.crop((694, 64, 828, 84))
    date_text = pytesseract.image_to_string(date, lang='eng')[:-2]
    return_text = [title_text, region_text, date_text]
    return return_text


def export_csv_data(fname, starts_from, cloud, temperature, rain):
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        date_start = datetime.datetime(starts_from[0], starts_from[1], starts_from[2], starts_from[3]) #year,month,day,hour

        for i in range(0,72):
            date_for_this_point = date_start+datetime.timedelta(hours=i)

            stamped_data = (date_start, date_for_this_point, cloud[i], temperature[i], rain[i])
            writer.writerow(stamped_data)


def export_easy_to_read_data(fname, starts_from, cloud, temperature, rain):
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["    datetime", "        cloud", "temperature", "rain"])
        max_digit = 1
        for accuracy in decimal_accuracy:
            if(accuracy > max_digit):
                max_digit = accuracy
        max_digit += 4 #xx. -> additional 3

        date_start = datetime.datetime(starts_from[0], starts_from[1], starts_from[2], starts_from[3]) #year,month,day,hour

        for i in range(0,72):
            date_for_this_point = date_start+datetime.timedelta(hours=i)
            data_string = [str(cloud[i]), str(temperature[i]), str(rain[i])]
            counter = 0
            for string in data_string:
               string = string + " "* ( (max_digit - len(string) ))
               data_string[counter] = string
               counter +=1
            stamped_data = (date_for_this_point, "   "+data_string[0] , data_string[1], data_string[2])
            writer.writerow(stamped_data)


def extract_meteogram_data(image):
    image_array = np.array(image)

    texts = extract_title_region_date(image)

    if(texts[0] != "WRF METEOGRAM"):
       raise ImageParseError("Image does not appear to be as expected. Text 'WRF METEOGRAM' not found.")

    date_array = get_date_as_number(texts[2]) ## year, month, day, shift

    cloud = extract_data(image_array, 0)

    temperature_pixels=extract_data(image_array, 1)
    temperature = initiliaze_temperature_pixels (image_array, image, temperature_pixels)

    rain = extract_data(image_array, 2)

    return date_array, cloud, temperature, rain


def main(argv=None):
    if argv is None:
        argv = sys.argv

    for fname in argv[1:]:
        file, ext = os.path.splitext(fname)
        logger.info("Processing file:" + fname)
        image = Image.open(fname)
        date_array, cloud, temperature, rain = extract_meteogram_data(image)
        export_csv_data(file+'.csv', date_array, cloud, temperature, rain)
        export_easy_to_read_data(file+'.txt', date_array, cloud, temperature, rain)


if __name__ == '__main__':
    sys.exit(main())


























