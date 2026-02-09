from data_provider.data_loader import airfoil, ns, darcy, pipe, elas, plas, pdebench_autoregressive, \
    pdebench_steady_darcy, car_design, cfd3d, car_design_upt


def get_data(args):
    data_dict = {
        'car_design_upt': car_design_upt,
        'car_design': car_design,
        'pdebench_autoregressive': pdebench_autoregressive,
        'pdebench_steady_darcy': pdebench_steady_darcy,
        'elas': elas,
        'pipe': pipe,
        'airfoil': airfoil,
        'darcy': darcy,
        'ns': ns,
        'plas': plas,
        'cfd3d': cfd3d,
    }
    dataset = data_dict[args.loader](args)
    train_loader, test_loader, shapelist = dataset.get_loader()
    return dataset, train_loader, test_loader, shapelist
