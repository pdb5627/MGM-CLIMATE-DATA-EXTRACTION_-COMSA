## Author: ERDEM CANAZ
## email : erdemcanaz@gmail.com
## Modifications by Paul Brown
## https://github.com/user/pdb5627
from typing import Optional
import string

from PIL import Image
import numpy as np
import numpy.ma as ma
import pytesseract
import os, sys
from collections import namedtuple
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageParseError(Exception):
    """ Raised when there is a problem getting data out of the meteogram image. """
    pass


# All plots are aligned vertically
left_edge = 75
right_edge = 825
horizontal_pixels = np.linspace(79, 820, 72).round().astype(int)

PlotAxis = namedtuple('PlotAxis', ['top', 'bot', 'left', 'right'])

cloud_axis = PlotAxis(118, 235, left_edge, right_edge)
temp_axis = PlotAxis(294, 529, left_edge, right_edge)
temp_legend = PlotAxis(temp_axis.top, temp_axis.top+20, temp_axis.left, temp_axis.left+30) # Legend area to mask out
rain_axis = PlotAxis(588, 706, left_edge, right_edge)

top_and_bottoms = [119,234,313,528,589,705] # cloud, temperature, rain
mm_per_mSquare_per_px = 0.0650 # if there is no catastrophic event, good approximation for ankara model!


def mean_true_idx(a):
    row_index_array = np.broadcast_to(np.arange(a.shape[0]), a.T.shape).T
    masked_array = np.ma.array(row_index_array, mask=~a)
    return masked_array.mean(axis=0)


def extract_data_new(image_array, ax: PlotAxis, masked: Optional[PlotAxis]=None, kind: str='line'):
    if masked is not None:
        # Make a copy so we don't wipe out the original data
        image_array = image_array.copy()
        image_array[masked.top:masked.bot, masked.left:masked.right, :] = 255  # Mask with white

    # Horizontal index based on what columns of the image we want to sample.
    # Sample lines three pixels left since the coordinates were calibrated to bar graphs, which show bars between
    # axis points.
    cropped_array = image_array[ax.top:ax.bot, horizontal_pixels - (3 if kind == 'line' else 0)].astype(int)
    # Image.fromarray(cropped_array.astype(np.uint8)).show()
    R = cropped_array[..., 0]
    G = cropped_array[..., 1]
    B = cropped_array[..., 2]

    is_gray = (np.abs(R - G) <= 10) & (np.abs(G - B) <= 10)

    # Bar charts: look for colored pixel closest to the top
    if kind == 'bar':
        yval_px = np.argmax(~is_gray, axis=0)

    # Line charts: find the average of the appropriately colored pixels at each x value.
    elif kind == 'line':
        color_match = ~is_gray & ((R - G > 30) | ((G > 160) & (R / B > 1.5)))
        yval_px = mean_true_idx(color_match)
    else:
        raise NotImplementedError(f"Unknown plot kind '{kind}'")

    yval_pu = 1 - yval_px / (ax.bot - ax.top - 1)
    return yval_pu


def read_yscale(image_array, image, ax):
    # Identify tick marks by color
    cropped_array = image_array[ax.top:ax.bot, ax.left+5].astype(int)
    R = cropped_array[..., 0]
    G = cropped_array[..., 1]
    B = cropped_array[..., 2]
    ticks = (abs(R - G) <= 10) & (abs(G - B) <= 10) & (R < 250)

    vertical_candidates = []
    values_of_temp_scale_lines = []
    tbl = str.maketrans('O', '0', ',%!$' + string.ascii_lowercase + string.ascii_uppercase)
    for tick in np.argwhere(ticks):
        line_y_coordinate = tick.item() + ax.top

        temp_scale = image.crop((ax.left - 35, line_y_coordinate - 10, ax.left - 1, line_y_coordinate + 10))
        # Upsampling the image seems to help tesseract recognize numbers better
        temp_scale = temp_scale.resize([2*d for d in temp_scale.size])
        # PSM modes: https://github.com/tesseract-ocr/tesseract/issues/434
        temp_scale_text = pytesseract.image_to_string(temp_scale, config='--psm 7')
        temp_scale_text = temp_scale_text.strip().translate(tbl)
        try:
            values_of_temp_scale_lines.append(float(temp_scale_text))
        except ValueError:
            continue
        vertical_candidates.append(line_y_coordinate)

    # Sanity check on values. Values go top to bottom, so they should be decreasing.
    pts = (0, len(values_of_temp_scale_lines) - 1)  # Default to first and last point
    if values_of_temp_scale_lines[0] <= values_of_temp_scale_lines[1] \
            and len(values_of_temp_scale_lines) > 2 \
            and values_of_temp_scale_lines[1] > values_of_temp_scale_lines[2]:
        # Point 0 is bogus, use Points 1 and 2
        pts = (1, 2)

    dt = values_of_temp_scale_lines[pts[0]] - values_of_temp_scale_lines[pts[1]]
    dpx = vertical_candidates[pts[0]] - vertical_candidates[pts[1]]
    slope = dt / dpx
    # Another sanity check
    if slope > 0:
        raise ImageParseError(f'Scale is off. px = ({vertical_candidates[pts[0]]}, {vertical_candidates[pts[1]]}), '
                              f'y = ({values_of_temp_scale_lines[pts[0]]}, {values_of_temp_scale_lines[pts[1]]})')
    ax_vals = []
    for px in (ax.top, ax.bot):
        val = (px - vertical_candidates[pts[0]]) * slope + values_of_temp_scale_lines[pts[0]]  # line equation
        ax_vals.append(val)
    return ax_vals


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


def scale_temperature_pixels(image_array, image, temperature_pixel_data_array):
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
            # Upsampling the image seems to help tesseract recognize numbers better
            temp_scale = temp_scale.resize([2 * d for d in temp_scale.size])
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
    # Another sanity check
    if slope > 0:
        raise ImageParseError(f'Scale is off. px = ({vertical_candidates[pts[0]]}, {vertical_candidates[pts[1]]}), '
                              f'y = ({values_of_temp_scale_lines[pts[0]]}, {values_of_temp_scale_lines[pts[1]]})')
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

    cloud = np.array(extract_data(image_array, 0))
    cloud2 = extract_data_new(image_array, cloud_axis, None, 'bar')
    cloud_diff = cloud - cloud2
    if cloud_diff.max() > 0.1:
        logger.warning(f'Unusually large cloud diff. ({cloud_diff.max()} at h={cloud_diff.argmax()})')

    temperature_pixels=extract_data(image_array, 1)
    temperature = np.array(scale_temperature_pixels(image_array, image, temperature_pixels))
    temp2_pu = extract_data_new(image_array, temp_axis, temp_legend, 'line')
    temp_scale = read_yscale(image_array, image, temp_axis)
    temp2_real = temp2_pu*(temp_scale[0] - temp_scale[1]) + temp_scale[1]
    temp_diff = temperature - temp2_real
    if temp_diff.max() > 0.5:
        logger.warning(f'Unusually large temperature diff. ({temp_diff.max()} at h={temp_diff.argmax()})')

    rain = extract_data(image_array, 2)
    rain2 = extract_data_new(image_array, rain_axis, None, 'bar')
    rain2_real = rain2*mm_per_mSquare_per_px*(rain_axis.bot - rain_axis.top)
    rain_diff = rain - rain2_real
    if rain_diff.max() > 0.05:
        logger.warning(f'Unusually large rainfall diff. ({rain_diff.max()} at h={rain_diff.argmax()})')

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


























