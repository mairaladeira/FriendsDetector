__author__ = 'Maira'
import sys
import os.path

# Script for creating a csv file from a directory tree hierarchy
#
#
#  .
#  |-- README
#  |-- c1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- c2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- c3
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print "usage: create_csv <base_path> <destination_file>"
        sys.exit(1)

    BASE_PATH=sys.argv[1]
    destination_file = sys.argv[2]
    SEPARATOR=";"
    file_text = ""
    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                if ".DS_Store" not in filename:
                    print filename
                    abs_path = "%s/%s" % (subject_path, filename)
                    file_text += "%s%s%d" % (abs_path, SEPARATOR, label)
                    file_text += "\n"
            #print "%s%s%d" % (abs_path, SEPARATOR, label)
            label = label + 1
    file = open("data/"+destination_file, 'w+')
    file.write(file_text)
    file.close()
    print(file_text)