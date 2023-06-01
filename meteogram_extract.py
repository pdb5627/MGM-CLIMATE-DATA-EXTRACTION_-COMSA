# Author: ERDEM CANAZ
# email : erdemcanaz@gmail.com
# Modifications by Paul Brown
# https://github.com/user/pdb5627
from typing import Optional
import string

from PIL import Image
import numpy as np
import pytesseract
import os
import sys
from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


class ImageParseError(Exception):
    """ Raised when there is a problem getting data out of the meteogram image. """
    pass


# All plots are aligned vertically
left_edge = 75
right_edge = 825
horizontal_pixels_bar = np.linspace(79, 820, 72).round().astype(int)
horizontal_pixels_line = np.linspace(76, 826, 72).round().astype(int)
horizontal_pixels_line[-1] = right_edge - 1

PlotAxis = namedtuple('PlotAxis', ['top', 'bot', 'left', 'right'])

cloud_axis = PlotAxis(118, 235, left_edge, right_edge)
temp_axis = PlotAxis(294, 529, left_edge, right_edge)
temp_legend = PlotAxis(temp_axis.top, temp_axis.top+20, temp_axis.left, temp_axis.left+30)  # Legend area to mask out
rain_axis = PlotAxis(588, 706, left_edge, right_edge)

mm_per_mSquare_per_px = 0.0650  # if there is no catastrophic event, good approximation for ankara model!


def mean_true_idx(a, axis=0):
    """
    Given an array with True and False values, returns the average index of these true values along the specified axis.
    :param a: Numpy array of boolean values
    :param axis: Axis along which to calculate the mean indices
    :return: Numpy (masked) array with one less dimension than a
    """
    row_index_array = np.broadcast_to(np.arange(a.shape[0]), a.T.shape).T
    masked_array = np.ma.array(row_index_array, mask=~a)
    return masked_array.mean(axis=axis)


def extract_data_new(image_array, ax: PlotAxis, masked: Optional[PlotAxis] = None, kind: str = 'line'):
    if masked is not None:
        # Make a copy so we don't wipe out the original data
        image_array = image_array.copy()
        image_array[masked.top:masked.bot, masked.left:masked.right, :] = 255  # Mask with white

    # Horizontal index based on what columns of the image we want to sample.
    # Sample lines three pixels left since the coordinates were calibrated to bar graphs, which show bars between
    # axis points.
    cropped_array = image_array[ax.top:ax.bot, horizontal_pixels_line if kind == 'line' else
                                               horizontal_pixels_bar].astype(int)
    # Image.fromarray(cropped_array.astype(np.uint8)).show() # Use for troubleshooting
    R = cropped_array[..., 0]
    G = cropped_array[..., 1]
    B = cropped_array[..., 2]

    is_gray = (np.abs(R - G) <= 10) & (np.abs(G - B) <= 10)

    # Bar charts: look for colored pixel closest to the top
    if kind == 'bar':
        yval_px = np.argmax(~is_gray, axis=0)

    # Line charts: find the average of the appropriately colored pixels at each x value.
    elif kind == 'line':
        color_match = ~is_gray & ((R - G > 30) | ((G > 160) & (R / (B+1) > 1.5)))
        yval_px = mean_true_idx(color_match)
    else:
        raise NotImplementedError(f"Unknown plot kind '{kind}'")

    yval_pu = 1 - yval_px / (ax.bot - ax.top - 1)
    return yval_pu


def read_yscale(image_array, image, ax):
    # Identify tick marks by color and darkness
    cropped_array = image_array[(ax.top+1):ax.bot, (ax.left+1):(ax.left+5)].astype(int)
    R = cropped_array[..., 0]
    G = cropped_array[..., 1]
    B = cropped_array[..., 2]
    ticks = (abs(R - G) <= 10) & (abs(G - B) <= 10) & (R < 250) & (G < 250)
    ticks = ticks.any(axis=1)

    vertical_candidates = []
    values_of_temp_scale_lines = []
    tbl = str.maketrans('O—', '0-', ',%!$' + string.ascii_lowercase + string.ascii_uppercase)
    for tick in np.argwhere(ticks):
        line_y_coordinate = tick.item() + ax.top + 1

        temp_scale = image.crop((ax.left - 35, line_y_coordinate - 10, ax.left - 1, line_y_coordinate + 10))
        # Upsampling the image seems to help tesseract recognize numbers better
        temp_scale = temp_scale.resize([2*d for d in temp_scale.size])
        if _log.getEffectiveLevel() <= 10:
            temp_scale.show()
        # PSM modes: https://github.com/tesseract-ocr/tesseract/issues/434
        temp_scale_text = pytesseract.image_to_string(temp_scale, config='--psm 7')
        temp_scale_text = temp_scale_text.strip().translate(tbl)
        try:
            prospective_value = float(temp_scale_text)
            _log.debug(f"{line_y_coordinate=}, {prospective_value=}")
        except ValueError:
            continue
        # Sometimes tesseract might ignore a minus sign.
        if values_of_temp_scale_lines and prospective_value > values_of_temp_scale_lines[-1] > -1*prospective_value:
            _log.debug(f"Inverting prospective value ({prospective_value} -> ({-prospective_value}))")
            prospective_value *= -1
        values_of_temp_scale_lines.append(prospective_value)
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
    # Another sanity check. Scale should have increasing values going up
    if slope > 0:
        raise ImageParseError(f'Scale is off. px = ({vertical_candidates[pts[0]]}, {vertical_candidates[pts[1]]}), '
                              f'y = ({values_of_temp_scale_lines[pts[0]]}, {values_of_temp_scale_lines[pts[1]]})')
    ax_vals = []
    for px in (ax.top, ax.bot):
        val = (px - vertical_candidates[pts[0]]) * slope + values_of_temp_scale_lines[pts[0]]  # line equation
        ax_vals.append(val)
    return ax_vals


def extract_title_region_date(image):
    title = image.crop((338, 16, 560, 41))
    title_text = pytesseract.image_to_string(title, lang='eng').strip()

    region = image.crop((84, 32, 149, 55))
    region_text = pytesseract.image_to_string(region, lang='eng').strip().replace('0', 'O')

    date = image.crop((694, 64, 828, 84))
    date_text = pytesseract.image_to_string(date, lang='eng',
                                            config='--psm 7 -c tessedit_char_whitelist="0123456789 /GMT"').strip()
    return title_text, region_text, date_text


def extract_meteogram_data(image):
    image_array = np.array(image)

    title_text, region_text, date_text = extract_title_region_date(image)

    if title_text != "WRF METEOGRAM":
        raise ImageParseError("Image does not appear to be as expected. Text 'WRF METEOGRAM' not found.")

    current_dt = pd.to_datetime(date_text, format='%d/%m/%Y %H%Z')
    dt = pd.date_range(current_dt, periods=72, freq='1H')

    cloud = extract_data_new(image_array, cloud_axis, None, 'bar')
    # No scaling applied to cloud since the y axis is always from 0 to 100%

    temp_pu = extract_data_new(image_array, temp_axis, temp_legend, 'line')
    temp_scale = read_yscale(image_array, image, temp_axis)
    temp_real = temp_pu*(temp_scale[0] - temp_scale[1]) + temp_scale[1]

    rain = extract_data_new(image_array, rain_axis, None, 'bar')
    # TODO: Detect scale automatically or (even better) read text above bars of precipitation chart
    # rather than hard-coding it like this.
    rain_real = rain*mm_per_mSquare_per_px*(rain_axis.bot - rain_axis.top)

    df = pd.DataFrame({'location': region_text,
                       'current_dt': current_dt,
                       'dt': dt,
                       'clouds': cloud,
                       'temperature': temp_real,
                       'rain': rain_real}, index=dt)

    return df


def process_file(fname: str, write_csv: Optional[bool] = False, plot: Optional[bool] = False):
    file, ext = os.path.splitext(fname)
    _log.info(f"Processing file: {fname}")
    image = Image.open(fname)
    df = extract_meteogram_data(image)
    if write_csv:
        df.to_csv(file + '.csv', index=False)

    if plot:
        fig = plt.figure(figsize=(5.3, 7.5))
        ax1 = fig.add_subplot(7, 1, (1, 2))
        ax2 = fig.add_subplot(7, 1, (3, 5))
        ax3 = fig.add_subplot(7, 1, (6, 7))
        fig.tight_layout(h_pad=3)
        plt.subplots_adjust(top=0.9)

        # TODO: Add current_dt to the figure somewhere
        ax1.bar(df.index, df['cloud']*100, align='edge', width=1/24*0.8, color='#b2b9f6', edgecolor='#4051e9')
        ax2.plot(df.index, df['temperature'], color='#FF000099')
        ax3.bar(df.index, df['rain'], align='edge', width=1/24*0.8, color='#b2b9f6', edgecolor='#4051e9')

        ax1.set_ylim(0, 100)
        ax3.set_ylim(0, 6.25)
        ax2.yaxis.grid(which='major', linestyle=':')
        # Due to edge alignment of bar chart, it needs an extra period at the right in order not to cut off the last bar
        ax1.set_xlim(df.index[0], df.index[-1] + pd.to_timedelta('59.9min'))
        ax2.set_xlim(df.index[0], df.index[-1])
        ax3.set_xlim(df.index[0], df.index[-1] + pd.to_timedelta('59.9min'))

        ax1.set_title('Cloud Cover (%)')
        ax2.set_title('Temperature (°C)')
        ax3.set_title('Hourly Rainfall (mm)')

        for ax in (ax1, ax2, ax3):
            ax.tick_params(which='both', direction='in')
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_minor_locator(mdates.HourLocator([6, 12, 18]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a'))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
            ax.xaxis.grid(which='major', linestyle='-')
            ax.xaxis.grid(which='minor', linestyle=':')
        plt.savefig(file + '_digitized.png')
        #plt.show()
        plt.close()
    return df


def main(argv=None):
    if argv is None:
        argv = sys.argv

    files_with_errors = []

    for fname in argv[1:]:
        try:
            df = process_file(fname, write_csv=True, plot=False)
        except SystemExit:
            break
        except Exception as e:
            _log.error(f"Failed to process {fname}.", exc_info=True)
            files_with_errors.append(fname)

    if files_with_errors:
        _log.info(f"Failures to process the following files:")
        for fname in files_with_errors:
            _log.info(fname)


if __name__ == '__main__':
    sys.exit(main())

