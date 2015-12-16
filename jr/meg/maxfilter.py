import numpy as np
import os


def maxfilter_tsss(data_path, files):
    # Read HPI from all files
    r = maxfilter_find_head_alignment(data_path, files)
    # execute maxfilter command for each run
    for f in files:
        source = data_path + f
        trans = data_path + files[r]
        print('Maxfilter ' + source)
        # run maxfilter
        os.system(
            '/neurospin/local/neuromag/bin/util/maxfilter -f ' + source +
            ' -origin 0.0 0.0 40.0 -frame head -autobad 60 -badlimit 7 ' +
            '-trans ' + trans)


def maxfilter_find_head_alignment(data_path, files):
    # Read HPI from all files
    XYZ = list()
    for f in files:
        cmd = os.popen(
            "show_fiff -vt 222 " + data_path + f + " | tail -n 1")
        xyz = [float(i) for i in cmd.read().replace(')', '').split()[4:7]]
        XYZ.append(xyz)

    print(XYZ)
    # Compute distance from the mean position for each run
    m = np.mean(XYZ, axis=0)
    distance = [sum((xyz_ - m) ** 2) for xyz_ in XYZ]

    # Find most representative run
    r = np.argmin(distance)
    print('Most representative run ' + files[r])
    return r
