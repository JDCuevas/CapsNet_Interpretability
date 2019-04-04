import os
import numpy as np
from h5py import File
from tqdm import tqdm
from networks import CAPSNET
from keras.utils import to_categorical as one_hot
from keras.models import Model
from argparse import ArgumentParser
from time import clock
from keras.utils import plot_model
from keras import backend as K

import sys
sys.path.insert(0, "utils")
from epochs import stop, write_log

seed = 1234

if __name__ == '__main__':
    parser = ArgumentParser(description = "Capsule network on protein volumetric data.")
    parser.add_argument('--epochs', default = 50, type = int)
    parser.add_argument('--filters', default = 256, type = int)
    parser.add_argument('--kernel_size', default = 9, type = int)
    parser.add_argument('--strides', default = 2, type = int)
    parser.add_argument('--primarycaps_dim', default = 8, type = int)
    parser.add_argument('--voxelcap_dim', default = 16, type = int)
    parser.add_argument('--dense_1_units', default = 512, type = int)
    parser.add_argument('--dense_2_units', default = 1024, type = int)

    parser.add_argument('--result_dir', default = 'capsnet_results/')
    parser.add_argument('--data_folder', default = '../../data/KrasHras/')
    parser.add_argument('--dim_type', default = '-2d')
    parser.add_argument('--debug', default = 0, type = int)
    parser.add_argument('--lr', default = 0.001, type = float)
    parser.add_argument('--lr_decay', default = 0.9, type = float)
    parser.add_argument('--routings', default = 3, type = int)
    parser.add_argument('--nb_chans', default = 8, type = int)
    parser.add_argument('--graph_model', default = 1, type = int)

    args = parser.parse_args()

    file_name = args.result_dir + CAPSNET.__name__ + ("_filters_%d_kernel_size_%d_strides_%d_primarycaps_dim_%d_voxelcap_dim_%d_dense_1_units_%d_dense_2_units_%d" % (args.filters, args.kernel_size, args.strides, args.primarycaps_dim, args.voxelcap_dim, args.dense_1_units, args.dense_2_units,))

    if bool(args.debug) != 1:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load HDF5 dataset
    f = File(args.data_folder + "dataset.hdf5", "r")

    # Shuffle train data
    train_set = f['train']
    classes = list(train_set.keys())

    x_train = []
    y_train = []
    for i in range(len(classes)):
        x = [name for name in train_set[classes[i]] if name.endswith(args.dim_type)]
        y = [i for j in range(len(x))]
        x_train += x
        y_train += y
    x_train = np.expand_dims(x_train, axis = -1)
    y_train = np.expand_dims(y_train, axis = -1)
    train = np.concatenate([x_train,y_train], axis = -1)
    np.random.seed(seed)
    np.random.shuffle(train)

    # Shuffle validation data
    val_set = f['val']
    classes = list(val_set.keys())

    x_val = []
    y_val = []
    for i in range(len(classes)):
        x = [name for name in val_set[classes[i]] if name.endswith(args.dim_type)]
        y = [i for j in range(len(x))]
        x_val += x
        y_val += y
    x_val = np.expand_dims(x_val, axis = -1)
    y_val = np.expand_dims(y_val, axis = -1)
    val = np.concatenate([x_val, y_val], axis = -1)
    np.random.seed(seed)
    np.random.shuffle(val)

    # Load Model
    print('Generating Model')
    model, eval_model = CAPSNET(args, len(classes))
    
    model.summary()

    if args.graph_model:
        plot_model(model, to_file = '%s_model_plot.png' % file_name, show_shapes = True, show_layer_names = True)

    # Training Loop
    history = []
    best_val_acc = 0.0
    # best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print("Epoch %d" % (epoch + 1))

        # Fit training data
        print('Fit training data')
        train_status = []
        train_time = 0.0
        
        for i in tqdm(range(len(train))):
            x = np.array(train_set[classes[int(train[i, 1])] + '/' + train[i, 0]])
            x = np.expand_dims(x, axis = 0)
            y = one_hot(train[i, 1], num_classes = len(classes))
            y = np.expand_dims(y, axis = 0)
            
            train_start_time = clock()
            output = model.train_on_batch([x, y], [y])
            train_time += clock() - train_start_time            
            train_status.append(output)

        # Calculate training loss and accuracy
        train_status = np.array(train_status)
        
        if args.debug == True:
            print(train_status)
        
        train_loss = np.average(train_status[:, 0])
        train_acc = np.average(train_status[:, 1])
        print('Train Loss ->', train_loss)
        print('Train Accu ->', train_acc)
        print('Train Time ->', train_time, '\n')

        # Test on validation data
        print('Testing on Validation Data')
        val_status = []
        val_time = 0.0
        for i in tqdm(range(len(val))):
            x = np.array(val_set[classes[int(val[i, 1])] + '/' + val[i, 0]])
            x = np.expand_dims(x, axis = 0)
            y = one_hot(val[i, 1], num_classes = len(classes))
            y = np.expand_dims(y, axis = 0)
            
            val_start_time = clock()
            output = model.test_on_batch([x, y], [y])
            val_time += clock() - val_start_time
            
            val_status.append(output)

        # Calculate validation loss and accuracy
        val_status = np.array(val_status)
        
        if args.debug == True:
            print(val_status)
        
        val_loss = np.average(val_status[:, 0])
        val_acc = np.average(val_status[:, 1])
        print('Val Loss ->', val_loss)
        print('Val Accu ->', val_acc)
        print('Val Time ->', val_time, '\n')

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            print('Saving Model Weights')
            model.save_weights(file_name + '_best_acc_loss_weights.hdf5')

        history.append([(epoch + 1), train_loss, train_acc, train_time, val_loss, val_acc, val_time])

        # if train_acc == 1:
        #     print("Stopping early")
        #     break

    # Parse test data
    test_set = f['test']
    classes = list(test_set.keys())

    x_test = []
    y_test = []
    for i in range(len(classes)):
        x = [name for name in test_set[classes[i]] if name.endswith(args.dim_type)]
        y = [i for j in range(len(x))]
        x_test += x
        y_test += y


    x_test = np.expand_dims(x_test, axis = -1)
    y_test = np.expand_dims(y_test, axis = -1)
    test = np.concatenate([x_test,y_test], axis = -1)

    # Load weights of best model
    model.load_weights(file_name + '_best_acc_loss_weights.hdf5')

    # Evaluate test data
    print('Evaluating Test Data')
    test_status = []
    test_time = 0.0

    for i in tqdm(range(len(test))):
        x = np.array(test_set[classes[int(test[i, 1])] + '/' + test[i, 0]])
        x = np.expand_dims(x, axis = 0)
        yy = int(test[i, 1])
        y = one_hot(test[i, 1], num_classes = len(classes))
        y = np.expand_dims(y, axis = 0)
    
        test_start_time = clock()
        output = eval_model.test_on_batch([x, y], [y])
        test_time += clock() - test_start_time
        
        test_status.append(output)

    # Calculate test loss and accuracy
    test_status = np.array(test_status)

    if args.debug:
        print(test_status)

    test_loss = np.average(test_status[:, 0])
    test_acc = np.average(test_status[:, 1])
    print('Test Loss ->', test_loss)
    print('Test Accu ->', test_acc)
    print('Test Time ->', test_time, '\n')

    # Save training history to csv file
    history = np.array(history)
    test_footer = 'Test [loss accu time], %f, %f, %f' % (test_loss, test_acc, test_time)
    
    np.savetxt(
        file_name + '_metrics.csv',
        history,
        fmt = '%1.3f',
        delimiter = ', ',
        header = 'epoch, train_loss, train_acc, train_time, val_loss, val_acc, val_time',
        footer = test_footer
    )

    sys.path.insert(0, "DeepExplain")
    from deepexplain.tensorflow import DeepExplain

    with DeepExplain(session=K.get_session()) as de:
        input_tensor = model.layers[0].input

        deModel = Model(inputs=input_tensor, output=model.layers[-1].output)
        target_tensor = deModel(input_tensor)

        x = np.array(test_set[classes[int(test[0, 1])] + '/' + test[0, 0]])
        x = np.expand_dims(x, axis = 0)
        y = one_hot(test[0, 1], num_classes = len(classes))
        y = np.expand_dims(y, axis = 0)


        xs = x
        ys = y

        attributions_gradin = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
        #attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys, batch_size=1)
        #attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
        #attributions_sv     = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=1)
        print ("Done")

    # Plotting util methods
    from skimage import feature, transform
    import matplotlib.pyplot as plt

    def plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8, channel=0):
        dx, dy = 0.05, 0.05
        xx = np.arange(0.0, data.shape[1], dx)
        yy = np.arange(0.0, data.shape[0], dy)
        xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        extent = xmin, xmax, ymin, ymax
        cmap_xi = plt.get_cmap('Greys_r')
        cmap_xi.set_bad(alpha=0)
        overlay = None
        if xi is not None:
            # Compute edges (to overlay to heatmaps later)
            xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=0) # Changed axis to 0 from -1 because my xi is in shape (1,512,512), not (512,512,1) 
            in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
            edges = feature.canny(in_image_upscaled).astype(float)
            edges[edges < 0.5] = np.nan
            edges[:5, :] = np.nan
            edges[-5:, :] = np.nan
            edges[:, :5] = np.nan
            edges[:, -5:] = np.nan
            overlay = edges

        abs_max = np.percentile(np.abs(data), percentile)
        abs_min = abs_max

        if len(data.shape) == 3:
            data = np.mean(data, 2)
        axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
        if overlay is not None:
            axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
        axis.axis('off')
        plt.savefig('attributions/{}.png'.format('attributions_channel_' + str(channel)))
        return axis

    n_cols = 2
    n_rows = 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*10, n_rows*10))

    for i in range(xs.shape[3]):
        plot(xs[:,:,:,i].reshape(512,512), cmap='Greys', axis=axes[0], channel=i).set_title('Original')
        plot(attributions_gradin[:,:,:,i].reshape(512,512), xi = xs[:,:,:,i], axis=axes[1], channel=i).set_title('Grad*Input')

    # NOTES ON PLOTTING: I removed the for loop because in the Deep Explain example, the guy feeds his de model 10 samples from his test set. I however
    #                    only feed the deep explain model a single sample. So in the example, the shape of xs is (10, 28, 28, 1) because there are 10 
    #                    samples, my xs is (1, 512, 512, 8), a single example (the 1 in my xs doesn't refer to this, its the same 1 in his xs) with 8 channels.



    '''n_cols = 6
                n_rows = int(len(attributions_gradin) / 2)
                fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3*n_cols, 3*n_rows))
            
                for i, a1 in enumerate(attributions_gradin):
                    row, col = divmod(i, 2)
                    plot(a1[:,:,i].reshape(512.512), cmap='Greys', axis=axes[row, col*3]).set_title('Original')
                    plot(a1.reshape(512, 512), xi = xs[i], axis=axes[row,col*3+1]).set_title('Grad*Input')
            '''
    '''
    for i, (a1, a2) in enumerate(zip(attributions_gradin, attributions_sv)):
        row, col = divmod(i, 2)
        plot(xs[i].reshape(512, 512), cmap='Greys', axis=axes[row, col*3]).set_title('Original')
        plot(a1.reshape(512, 512), xi = xs[i], axis=axes[row,col*3+1]).set_title('Grad*Input')
        plot(a2.reshape(512, 512), xi = xs[i], axis=axes[row,col*3+2]).set_title('Shapley Values')
    '''