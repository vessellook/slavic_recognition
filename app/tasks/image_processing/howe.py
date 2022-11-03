"""An implementation of Howe binarization for document images.

Taken from https://github.com/duerig/laser-dewarp/tree/master/process/binarize
"""

import maxflow, argparse, cv2, math, numpy, os, sys
from scipy import signal, ndimage


__all__ = ['binarize']


# Taken from http://wiki.scipy.org/Cookbook/SignalSmooth
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


def hysteresis(absg, suppress, thi, tlo, allow=None):
    if suppress is not None:
        absg = numpy.where(suppress, 0, absg)
    absmax = numpy.amax(absg[1:-1, 1:-1])
    high = (absg >= absmax * thi)
    low = numpy.logical_and(absg >= absmax * tlo,
                            absg < absmax * thi)
    close_kernel = numpy.asarray([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]])
    close = signal.convolve2d(high, close_kernel)[1:-1, 1:-1]
    seedY, seedX = numpy.nonzero(numpy.logical_and(low, close))
    if allow is not None:
        # high = numpy.logical_and(high, allow)
        low = numpy.logical_and(low, allow)
    for i in range(0, len(seedY)):
        floodfill(seedX[i], seedY[i], high, low)
    return high


def floodfill(startX, startY, dest, src):
    queue = [(startX, startY)]
    while len(queue) > 0:
        centerX, centerY = queue[-1]
        queue = queue[:-1]
        for x in range(centerX - 1, centerX + 2):
            for y in range(centerY - 1, centerY + 2):
                if 0 <= y < src.shape[0] and 0 <= x < src.shape[1] and src[y, x]:
                    dest[y, x] = 1
                    src[y, x] = 0
                    queue.append((x, y))


def image_cut(source, sink, horizontal, vertical, c):
    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes(source.shape)
    hStructure = numpy.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]])
    vStructure = numpy.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]])
    g.add_grid_edges(nodeids, weights=horizontal * c,
                     structure=hStructure,
                     symmetric=True)
    g.add_grid_edges(nodeids, weights=vertical * c,
                     structure=vStructure,
                     symmetric=True)
    g.add_grid_tedges(nodeids, source, sink)
    g.maxflow()
    return g.get_grid_segments(nodeids)


def algorithm1(img, thi=0.5, tlo=0.1, sigma=0.6,
               clist=(80,), f=None, csearch=False, thin=False):
    # compute base binarizations and the stability curve
    bsd = numpy.zeros(len(clist))
    bimg = f(img, thi, tlo, sigma, clist, csearch=csearch, thin=thin)
    for ic in range(1, len(clist)):
        bsd[ic - 1] = numpy.sum(numpy.not_equal(bimg[ic], bimg[ic - 1])) / float(bimg[ic].size)

    # smooth stability curve
    if len(clist) > 1:
        d = smooth(bsd[:-1], 5)[2:-2]
    else:
        d = bsd[:-1]

    r = 0
    scr = None
    for i in range(0, d.size - 2):
        for j in range(i + 2, d.size):
            for k in range(i + 1, j):
                v = d[i] + d[j] - 2 * d[k]
                if scr is None or v > scr:
                    q = i
                    r = k
                    s = j
                    scr = v
    print('algorithm1 ' + str(thi) + ' weighted at ' + str(r) + ': ' + str(clist[r]))
    return bimg[r], clist[r]


def algorithm2(img, sigma=0.6, clist=None, tlo=0.1,
               thilist=(0.1, 0.6), f=None, iter=5, csearch=False, thin=False):
    diffs = []
    images = []
    previous = f(img, thilist[0], thilist[0] / 3.0, sigma, clist, csearch=csearch, thin=thin)[0]
    for i in range(1, iter + 1):
        thi = thilist[0] + (thilist[1] - thilist[0]) * i / float(iter)
        tlo = thi / 3.0
        current = f(img, thi, tlo, sigma, clist, csearch=csearch, thin=thin)[0]
        images.append(previous)
        diffs.append(numpy.sum(numpy.not_equal(current, previous)))
        previous = current
    diffs = numpy.asarray(diffs)
    diffs = smooth(diffs, 5)[2:-2]
    index = numpy.argmin(diffs)
    return images[index], clist[0], thilist[0] + (thilist[1] - thilist[0]) * index / float(iter)


#####################################################

def algorithm3(img, sigma=0.6, clist=None, tlo=0.1,
               thilist=(0.25, 0.5), f=None, csearch=False, thin=False):
    if clist is None:
        clist = numpy.exp(numpy.linspace(numpy.log(10),
                                         numpy.log(640), num=15))
    blo, clo = algorithm1(img, thilist[0], tlo, sigma, clist, f=f, csearch=csearch, thin=thin)
    bmid, cmid = algorithm1(img, numpy.mean(thilist), tlo,
                            sigma, clist, f=f, csearch=csearch, thin=thin)
    bhi, chi = algorithm1(img, thilist[1], tlo, sigma, clist, f=f, csearch=csearch, thin=thin)
    dlo = numpy.sum(numpy.not_equal(blo, bmid))
    dhi = numpy.sum(numpy.not_equal(bhi, bmid))

    if dlo < dhi:
        return blo, clo, thilist[0]
    else:
        return bhi, chi, thilist[1]


def find_background_mask(img, threshold=2.0):
    sr = 31
    img2 = (img - numpy.float_(cv2.GaussianBlur(img, (sr, sr), sr * 3, borderType=cv2.BORDER_REFLECT)))
    rms = numpy.sqrt(cv2.GaussianBlur(img2 * img2, (sr, sr), sr * 3, borderType=cv2.BORDER_CONSTANT))
    return (img2 / (rms + 0.000000001)) > threshold


def sort_range(low, high):
    if low < high:
        return [low, high]
    else:
        return [high, low]

# An implementation of Howe binarization for document images.

# References:

# A Laplacian Energy for Document Binarization, N. Howe.  International Conference on Document Analysis and Recognition, 2011.
# http://cs.smith.edu/~nhowe/research/pubs/divseg-icdar.pdf
#
# Document Binarization with Automatic Parameter Tuning, N. Howe.  To appear in International Journal of Document Analysis and Recognition. DOI: 10.1007/s10032-012-0192-x.
# http://cs.smith.edu/~nhowe/research/pubs/divseg-ijdar.pdf
#
# Matlab Code:  http://cs.smith.edu/~nhowe/research/code/

version = '1.0'


def binarize(image, sigma=0.6, crange=None, trange=(0.15, 0.6), csearch=False, thin=False):
    if csearch:
        a = 60
        b = 3000
        if crange is not None:
            crange = sort_range(crange[0], crange[1])
            a = crange[0]
            b = crange[1]
        clist = numpy.exp(numpy.linspace(numpy.log(a), numpy.log(b), num=25))
    else:
        clist = [300]
    trange = sort_range(trange[0], trange[1])
    result, c, thi = algorithm2(image, clist=clist,
                                       csearch=csearch,
                                       thilist=trange,
                                       sigma=sigma, iter=20,
                                       thin=thin,
                                       f=binarize_single)
    # print 'c=', c, 'thi=', thi
    return result


# Use convolution to get the difference between every pixel and a
# neighbor specified by the offset. Offset is (y, x) coordinate
def subtract_neighbor(image, offset):
    kernel = numpy.asarray([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
    kernel[1 + offset[0], 1 + offset[1]] = -1
    sign = signal.convolve2d(image, kernel)[1:-1, 1:-1]
    return sign


# Canny edge detection
# Based originally on: http://pythongeek.blogspot.com/2012/06/canny-edge-detection.html
#                      https://github.com/rishimukherjee/Canny-Python
def canny(image, thi=0.5, tlo=0.1, sigma=0.6):
    # Gaussian filter
    smoothed = ndimage.gaussian_filter(image, sigma)

    # Sobel/Scharr convolution
    kernelx = numpy.asarray([[-3, 0, 3],
                             [-10, 0, 10],
                             [-3, 0, 3]])
    kernely = numpy.asarray([[-3, -10, -3],
                             [0, 0, 0],
                             [3, 10, 3]])

    gx = signal.convolve2d(smoothed, kernelx)[1:-1, 1:-1]
    gy = signal.convolve2d(smoothed, kernely)[1:-1, 1:-1]
    gmag = numpy.hypot(gx, gy)
    # Reflect it along the x-access because positive is down
    gdir = -numpy.arctan2(gy, gx)

    # Non-maximum suppression
    is_e = numpy.logical_or(numpy.logical_and(gdir < math.pi / 8,
                                              gdir >= -math.pi / 8),
                            numpy.logical_or(gdir >= 7 * math.pi / 8,
                                             gdir < -7 * math.pi / 8))
    is_ne = numpy.logical_or(numpy.logical_and(gdir < 3 * math.pi / 8,
                                               gdir >= math.pi / 8),
                             numpy.logical_and(gdir >= -7 * math.pi / 8,
                                               gdir < -5 * math.pi / 8))
    is_n = numpy.logical_or(numpy.logical_and(gdir < 5 * math.pi / 8,
                                              gdir >= 3 * math.pi / 8),
                            numpy.logical_and(gdir >= -5 * math.pi / 8,
                                              gdir < -3 * math.pi / 8))
    is_nw = numpy.logical_not(
        numpy.logical_or(is_e, numpy.logical_or(is_ne, is_n)))
    suppress_e = numpy.logical_or(subtract_neighbor(gmag, (0, 1)) < 0,
                                  subtract_neighbor(gmag, (0, -1)) < 0)
    suppress_ne = numpy.logical_or(subtract_neighbor(gmag, (-1, 1)) < 0,
                                   subtract_neighbor(gmag, (1, -1)) < 0)
    suppress_n = numpy.logical_or(subtract_neighbor(gmag, (1, 0)) < 0,
                                  subtract_neighbor(gmag, (-1, 0)) < 0)
    suppress_nw = numpy.logical_or(subtract_neighbor(gmag, (-1, -1)) < 0,
                                   subtract_neighbor(gmag, (1, 1)) < 0)

    suppress = numpy.logical_or(numpy.logical_or(numpy.logical_or(
        numpy.logical_and(is_e, suppress_e),
        numpy.logical_and(is_ne, suppress_ne)),
        numpy.logical_and(is_n, suppress_n)),
        numpy.logical_and(is_nw, suppress_nw))

    # Line tracing
    return hysteresis(gmag, suppress, thi, tlo)


def binarize_single(image, thi=0.5, tlo=0.1, sigma=0.6, clist=(100,), csearch=True, thin=False):
    # Ensure image is grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = numpy.float_(image)

    # Compute Laplacian for source/sink weights
    lkernel = numpy.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
    laplacian = signal.convolve2d(image, lkernel)[2:-2, 2:-2]

    # Find edges and exclude them from adjacency weights
    edge_mask = canny(image, thi=thi, tlo=tlo, sigma=sigma)
    dx = image[:-1, 1:] - image[:-1, :-1]
    dy = image[1:, :-1] - image[:-1, :-1]

    if thin:
        hc = numpy.logical_not(
            numpy.logical_or(
                numpy.logical_and(edge_mask[:-1, :-1], dx < 0),
                numpy.logical_and(edge_mask[:-1, 1:], dx >= 0)))[1:-1, 1:-1]
        vc = numpy.logical_not(
            numpy.logical_or(
                numpy.logical_and(edge_mask[:-1, :-1], dy < 0),
                numpy.logical_and(edge_mask[1:, :-1], dy >= 0)))[1:-1, 1:-1]
    else:
        hc = numpy.logical_not(
            numpy.logical_or(
                numpy.logical_and(edge_mask[:-1, :-1], dx > 0),
                numpy.logical_and(edge_mask[:-1, 1:], dx <= 0)))[1:-1, 1:-1]
        vc = numpy.logical_not(
            numpy.logical_or(
                numpy.logical_and(edge_mask[:-1, :-1], dy > 0),
                numpy.logical_and(edge_mask[1:, :-1], dy <= 0)))[1:-1, 1:-1]

    # Find high confidence background pixels
    background_mask = find_background_mask(image, threshold=1.5)[1:-1, 1:-1]

    result = []
    for c in clist:
        # Set source/sink weights
        if not csearch:
            weights = ((laplacian < 0) * c * -0.2) + ((laplacian >= 0) * c * 0.2)
        else:
            weights = laplacian
        weights = numpy.where(background_mask, 500, weights)[:-1, :-1]
        source = 1500 - weights
        sink = 1500 + weights

        # Partition the graph
        cut = numpy.int_(image_cut(source, sink, hc, vc, c))
        if thin:
            cut = numpy.logical_and(
                numpy.logical_and(cut[:-1, :-1], cut[1:, :-1]), cut[:-1, 1:])

        result.append(cut * 255)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Binarize document images using the Howe algorithm. Will search for the proper thi and c parameters. Optionally, fixed source/sink weights can be used instead of searching for c values to make it run faster. The resulting document will be very slightly smaller than the input document (3 pixels in each direction)')
    parser.add_argument('--version', action='version',
                        version='%(prog)s Version ' + version,
                        help='Get version information')
    parser.add_argument('--find-c', dest='find_c', default=False,
                        action='store_const', const=True,
                        help='Use variable weights for source/sink and search for the appropriate adjacency weight c. This is much slower but may yield better results.')
    parser.add_argument('--thin', dest='thin', default=False,
                        action='store_const', const=True,
                        help='Bias results to thin out letters. Reduces accuracy but improves readability.')
    parser.add_argument('--min-c', dest='min_c', default=60, type=int,
                        help='When searching for c, this is the minimum c value to look for. Defaults to 60')
    parser.add_argument('--max-c', dest='max_c', default=3000, type=int,
                        help='When searching for c, this is the maximum c value to look for. Defaults to 3000')
    parser.add_argument('--sigma', dest='sigma', default=0.6, type=float,
                        help='The level of smoothing done on the image before trying to find edges. Higher values reduce noise but may miss genuine edges. Defaults to 0.6')
    parser.add_argument('--min-thi', dest='min_thi', default=0.15, type=float,
                        help='Lowest thi value to search for during Canny edge detection. In each iteration, tlo is set to 1/3 of this value. Defaults to 0.15. Must be between 0 and 1.')
    parser.add_argument('--max-thi', dest='max_thi', default=0.6, type=float,
                        help='Highest thi value to search for during Canny edge detection. In each iteration, tlo is set to 1/3 of this value. Defaults to 0.6. Must be between 0 and 1.')
    parser.add_argument('input_file',
                        help='Path to input image file.')
    parser.add_argument('output_file',
                        help='Path to output image file.')
    options = parser.parse_args()
    if not os.path.exists(options.input_file):
        sys.stderr.write('howe: File not found: ' + options.input_file)
        exit(1)
    image = cv2.imread(options.input_file)
    result = binarize(image, sigma=options.sigma,
                      crange=[options.min_c, options.max_c],
                      trange=[options.min_thi, options.max_thi],
                      csearch=options.find_c,
                      thin=options.thin)
    cv2.imwrite(options.output_file, result)


if __name__ == '__main__':
    main()
