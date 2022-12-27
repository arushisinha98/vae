os.chdir("/u/paige/asinha/projectdir/")
with open('data-dump.csv') as file_name:
    data = np.loadtxt(file_name, delimiter = " ")
x_raw = np.asarray(data)

def periodic_padding(array, pad):
    N = len(array)
    M = N + 2*pad
    output = np.zeros(M)
    for index in range(pad, N+pad):
        output[index] = array[index - pad]
    for index in range(0, pad):
        output[index] = array[index + N - pad]
        output[index + N + pad] = array[index]
    return output

print("Size of x_raw: ", x_raw.shape)

if x_raw.shape[1] == 128:
    """ if already periodic padded, use x_raw to train """
    x_train = x_raw

if x_raw.shape[1] == 120:
    """ loop to pad x_raw and to store it as x_train """
    x_train = []
    for ix, xx in enumerate(x_raw):
        xx = periodic_padding(xx, 4)
        x_train.append(xx.astype('float32'))
    x_train = np.asarray(x_train)
