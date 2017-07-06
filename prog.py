import argparse
import csv
import datetime as dt
import os
import sys
import cv2

from scipy import ndimage
from sympy.physics.secondquant import wicks
from vector import pnt2line2
from skimage import color
from skimage.measure import label
from skimage.measure import regionprops
import numpy as np
import math
from matplotlib.pyplot import cm
from Crypto.PublicKey.pubkey import pubkey
from sklearn.datasets import fetch_mldata
from matplotlib.pyplot import cm
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage.morphology import *
import itertools
from skimage import color
import numpy as np
from skimage import color

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'excasa_lib'))

from excasa_lib import (
    detect_file_encoding, get_source_filenames, get_full_path,
    gen_dest_filename, info, warn, error
)

DEFAULT_DIR_NAME = 'non_ascii_results'
DEST_FILENAME_POSTFIX = 'NON_ASCII'
html_tags_map = {
    '<p>': '',
    '<br />': '',
    '<br/>': '',
    '<span>': '',
    '</span>': '',
    '</p>': '.',
}


def remove_new_line(text):
    return text.replace('\n', '')


def remove_non_ascii(text):
    return text.encode('ascii', errors='ignore').decode('utf-8')


def remove_html_tags(text):
    result = text

    for tag, replacement in html_tags_map.items():
        result = result.replace(tag, replacement)
    return result

	from excasa_lib import (
    info, debug, warn, error, critical,
    add_source_argument, add_destination_argument, CsvFilesWorker,
)


class NameSplittingCsvWorker(CsvFilesWorker):
    DEST_FILENAME_POSTFIX = 'NAME_SPLIT'
    DEFAULT_DIR_NAME = 'name_splitting_result'
    ADD_POSTFIX = False
    SCRIPT_DIR = PROJECT_DIR

    def __init__(self, fullname_col_name, **kw):
        super().__init__(**kw)
        self.fullname_col_name = fullname_col_name

    def parse_fullname(self, fullname):
        first_name = middle_name = last_name = ''
        fullname = fullname.strip().strip('-').strip(',').strip(
            '-').strip().strip(',')

        try:
            if fullname:
                parts = fullname.split(',')
                unusual = False

                if len(parts) != 1:
                    unusual = True

                parts = parts[0]  # parts after "," usually are useless
                parts = parts.split()

                if len(parts) >= 3:
                    first_name = parts[0]
                    middle_name = ' '.join(parts[1:-1])
                    last_name = parts[-1]
                elif len(parts) == 2:
                    first_name, last_name = parts
                else:
                    first_name = parts[0]

                if unusual:
                    debug('Fullname: %s' % fullname)
                    debug('Parts: {}, {}, {}\n'.format(
                        first_name, middle_name, last_name
                    ))
        except Exception as err:
            error('[FULL NAME PARSING ERROR]: "%s"' % fullname, exc_info=True)
        return first_name, middle_name, last_name

    def process_file(self, source_csv, dest_csv, total_lines):
        source_csv_headers = next(source_csv)

        if self.fullname_col_name not in source_csv_headers:
            info(
                '[COLUMN ERROR] The column "{}" not found in '
                'the file "{}". Skipping it.'.format(
                    self.fullname_col_name, self._current_source_file.name
                ))
            return

        fullname_col_index = source_csv_headers.index(self.fullname_col_name)
        dest_csv_headers = source_csv_headers.copy()
        dest_csv_headers.insert(fullname_col_index + 1, 'First Name')
        dest_csv_headers.insert(fullname_col_index + 2, 'Middle Name')
        dest_csv_headers.insert(fullname_col_index + 3, 'Last Name')

        new_lines = [dest_csv_headers]
        count = 1

        for line in source_csv:
            count += 1

            if count % 1000 == 0:
                dest_csv.writerows(new_lines)
                new_lines = []

                if count % 10000 == 0:
                    info('{} / {}'.format(count, total_lines))

            fullname = line[fullname_col_index]
            (first_name,
             middle_name,
             last_name) = self.parse_fullname(fullname)

            new_line = line.copy()
            new_line.insert(fullname_col_index + 1, first_name)
            new_line.insert(fullname_col_index + 2, middle_name)
            new_line.insert(fullname_col_index + 3, last_name)
            new_lines.append(new_line)

        dest_csv.writerows(new_lines)
        info('"{}" DONE. {} rows processed.'.format(
            self._current_source_file.name, count
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for name splitting')
    add_source_argument(parser)
    add_destination_argument(parser)
    parser.add_argument(
        '-c', '--column_name',
        default='User Name',
        type=str,
        help='name of the column with full name'
    )
    args = parser.parse_args()

    worker = NameSplittingCsvWorker(
        source=args.source, dest=args.dest, fullname_col_name=args.column_name
    )
    worker.start()


def main(source_path, dest_path, add_postfix):
    source_is_file = os.path.isfile(source_path)
    files_to_process = get_source_filenames(source_path)
    created_files = []

    for i, source_file_path in enumerate(files_to_process, start=1):
        info('File #{} out of {}'.format(i, len(files_to_process)))
        encoding, total_lines = detect_file_encoding(
            source_file_path, count=True
        )
        dest_filename = gen_dest_filename(
            source_is_file=source_is_file,
            source_file_path=source_file_path,
            dest_path=dest_path,
            postfix=DEST_FILENAME_POSTFIX,
            add_postfix=add_postfix,
            default_dir_name=DEFAULT_DIR_NAME,
        )

        if os.path.exists(dest_filename):
            info('"%s" already exists. Exit' % dest_filename)
            sys.exit()

        with open(source_file_path, encoding=encoding) as source_file:
            source_csv = csv.reader(source_file)
            dest_dir = os.path.split(dest_filename)[0]
            os.makedirs(dest_dir, exist_ok=True)

            with open(dest_filename, 'w', encoding='utf-8', newline='') as dest_file:
                created_files.append(dest_filename)
                j = 0
                changes = 0
                new_rows = []
                dest_csv = csv.writer(dest_file)
                csv_headers = next(source_csv)
                dest_csv.writerow(csv_headers)

                for j, row in enumerate(source_csv, start=1):
                    new_row = []

                    for cell_text in row:
                        new_text = remove_new_line(
                            remove_non_ascii(
                                remove_html_tags(cell_text)
                            )
                        )
                        new_row.append(new_text)

                        if new_text != cell_text:
                            changes += 1

                    new_rows.append(new_row)

                    if j % 1000 == 0:
                        dest_csv.writerows(new_rows)
                        new_rows = []

                        if j % 10000 == 0:
                            print('{} / {}'.format(j, total_lines))

                dest_csv.writerows(new_rows)
                print('%s lines processed.' % j)

        info('The file "%s" is processed.' % source_file_path)
        info('%s cells were changed.' % changes)

    info('All files processed.')

    if created_files:
        info('Were created files:')

        for file_name in created_files:
            info(file_name)
    else:
        warn('No files were created.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'source',
        action='store',
        type=str,
        help=('Source .csv file or folder, where all .csv files will '
              'be processed')
    )
    parser.add_argument(
        '-d', '--dest',
        default='',
        type=str,
        help=('Destination file or folder, optional. If not set, '
              'the name will be automatically generated including timestamp')
    )
    parser.add_argument(
        '--add-postfix', dest='add_postfix', action='store_const', const=True,
        default=False, help=(
            'Add or not postfix with timestamp to generated file names'
    ))
    args = parser.parse_args()
    source_path = get_full_path(args.source, main_path=PROJECT_DIR)
    dest_path = args.dest

    if not os.path.exists(source_path):
        error('The specified source "%s" does not exist. Exit.' % source_path)
        sys.exit()

    if os.path.exists(dest_path) and os.path.isfile(dest_path):
        error('The destination file "%s" already exists. Exit.' % dest_path)
        sys.exit()

    main(source_path, dest_path, add_postfix=args.add_postfix)
	
	
	import time


from skimage.morphology import *
from skimage import color
import numpy as np


videoName="data/video-2.avi"
cap = cv2.VideoCapture(videoName)


global x1
global y1
global linesForF
sumaBr = 0
    

def houghTransformtion(frame,grayImg,minLineLength,maxLineGap):
    edges = cv2.Canny(grayImg,50,150,apertureSize = 3)
    cv2.imwrite('lineDetected13.jpg',frame)# proveri da l treba
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength, maxLineGap)
    
    minx=9999
    miny=9999
    maxy=-2
    maxx=-2
    minx,miny,maxx,maxy=findPoints(lines)
    cv2.line(frame, (minx,miny), (maxx, maxy), (233, 0, 0), 2)
    return minx,miny,maxx,maxy

    def drawPoints(imget,pointArr):
    cv2.circle(imget, (pointArr[0]), 4, (25, 25, 255), 1)
    cv2.circle(imget, (pointArr[1]), 4, (25, 25, 255), 1)
    
def drawAllPoints(imget,linesAll):
    s=20
    for i in  range(len(linesAll)):
        for x1, y1, x2, y2 in linesAll[i]:
            cv2.circle(imget, (line[0]), 4, (2, 2, s), 1)
            cv2.circle(imget, (line[1]), 4, (2, 2, s), 1)
            s=s+20
    
def cheeseshop(kind, *arguments, **keywords):
    print "-- Do you have any", kind, "?"
    print "-- I'm sorry, we're all out of", kind
    for arg in arguments:
        print arg
    print "-" * 40
    keys = sorted(keywords.keys())
    for kw in keys:
        print kw, ":", keywords[kw]

def myFunct(cap):
    gray="grayFrame"  
    frame1="frame"
    kernel = np.ones((2,2),np.uint8)
    
    i=0
    if i==0:
        i=i+1
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAYY
                img0 = cv2.dilate(gray, kernel)
                frame1=frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break
    return frame1,img0
    
def findLineParams(videoName):
    cap = cv2.VideoCapture(videoName)
    maxL=610
    gap=9
    g=""
    f=""
    f,g=myFunct(cap)
    print("Start")
    print(cv2.__version__)
    cap.release()
    cv2.destroyAllWindows()
    return houghTransformtion(f,g,maxL,gap)

dd = -1
def convertIMG(img):
    
    img_BW=img==1.0
    return img_BW
    
def nextId():
    global dd
    dd += 1
    return dd

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        
        if(mdist<r):
            #print "distanca***** " + format(mdist)
            retVal.append(obj)

            
    return retVal

	def getDigit(img):
    img_BW=color.rgb2gray(img) >= 0.88
 
    img_BW=(img_BW).astype('uint8')

    newImg=putInLeftCorner(img_BW)
    i=0;
    minSum = 9000
    rez = -1
    while i<70000:
         sum=0
         if sum >= minSum:
             var a=0
         else
             minSum = sum
             rez = mnist.target[i]
         i=i+1
    return rez;

new_mnist_set=[]

BOARD_SIZE = 8
