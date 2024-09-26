import protonets.data
import protonets.data.custom_data_loader

def load(opt, splits):
    if opt['data.dataset'] == 'omniglot':
        ds, class_names = protonets.data.omniglot.load(opt, splits)
    elif opt['data.dataset'] == 'new_dataset': #opt['data.data_dir'] == 'new_dataset2'
        ds, class_names = protonets.data.custom_data_loader.load_data(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds, class_names
