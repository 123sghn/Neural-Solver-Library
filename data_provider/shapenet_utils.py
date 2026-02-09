import torch
import vtk
import os
import itertools
import random
import numpy as np
from torch_geometric import nn as nng
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph
from vtk.util.numpy_support import vtk_to_numpy


def load_unstructured_grid_data(file_name):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output


def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter


def get_sdf(target, boundary):
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def get_normal(unstructured_grid_data):
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(unstructured_grid_data)
    # visualize_poly_data(poly_data, surface_filter)
    # poly_data.GetPointData().SetScalars(None)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    # normal_filter.SetSplitting(0)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    '''
    normal_filter.SetComputeCellNormals(0)
    normal_filter.SetComputePointNormals(1)
    normal_filter.Update()
    #visualize_poly_data(poly_data, surface_filter, normal_filter)
    poly_data.GetPointData().SetNormals(normal_filter.GetOutput().GetPointData().GetNormals())
    p2c = vtk.vtkPointDataToCellData()
    p2c.ProcessAllArraysOn()
    p2c.SetInputData(poly_data)
    p2c.Update()
    unstructured_grid_data.GetCellData().SetNormals(p2c.GetOutput().GetCellData().GetNormals())
    #visualize_poly_data(poly_data, surface_filter, p2c)
    '''

    unstructured_grid_data.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    c2p = vtk.vtkCellDataToPointData()
    # c2p.ProcessAllArraysOn()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    # return unstructured_grid_data
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    # print(np.max(np.max(np.abs(normal), axis=1)), np.min(np.max(np.abs(normal), axis=1)))
    normal /= (np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)  # re-calculate
    # print(normal)
    return normal


def visualize_poly_data(poly_data, surface_filter, normal_filter=None):
    if normal_filter is not None:
        mask = vtk.vtkMaskPoints()
        mask.SetInputData(normal_filter.GetOutput())
        # mask.RandomModeOn()
        mask.Update()
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(mask.GetOutput())
        glyph.SetSourceData(arrow.GetOutput())
        glyph.SetVectorModeToUseNormal()
        glyph.SetScaleFactor(0.1)
        glyph.Update()
        norm_mapper = vtk.vtkPolyDataMapper()
        norm_mapper.SetInputData(normal_filter.GetOutput())
        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputData(glyph.GetOutput())
        norm_actor = vtk.vtkActor()
        norm_actor.SetMapper(norm_mapper)
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.GetProperty().SetColor(1, 0, 0)
        norm_render = vtk.vtkRenderer()
        norm_render.AddActor(norm_actor)
        norm_render.SetBackground(0, 1, 0)
        glyph_render = vtk.vtkRenderer()
        glyph_render.AddActor(glyph_actor)
        glyph_render.AddActor(norm_actor)
        glyph_render.SetBackground(0, 0, 1)

    scalar_range = poly_data.GetScalarRange()

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(surface_filter.GetOutputPort())
    mapper.SetScalarRange(scalar_range)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Set background to white

    renderer_window = vtk.vtkRenderWindow()
    renderer_window.AddRenderer(renderer)
    if normal_filter is not None:
        renderer_window.AddRenderer(norm_render)
        renderer_window.AddRenderer(glyph_render)
    renderer_window.Render()

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderer_window)
    interactor.Initialize()
    interactor.Start()


def get_datalist(root, samples, norm=False, coef_norm=None, savedir=None, preprocessed=False):
    dataset = []
    mean_in, mean_out = 0, 0
    std_in, std_out = 0, 0
    for k, s in enumerate(samples):
        if preprocessed and savedir is not None:
            save_path = os.path.join(savedir, s)
            if not os.path.exists(save_path):
                continue
            init = np.load(os.path.join(save_path, 'x.npy'))
            target = np.load(os.path.join(save_path, 'y.npy'))
            pos = np.load(os.path.join(save_path, 'pos.npy'))
            surf = np.load(os.path.join(save_path, 'surf.npy'))
            edge_index = np.load(os.path.join(save_path, 'edge_index.npy'))
        else:
            file_name_press = os.path.join(root, os.path.join(s, 'quadpress_smpl.vtk'))
            file_name_velo = os.path.join(root, os.path.join(s, 'hexvelo_smpl.vtk'))

            if not os.path.exists(file_name_press) or not os.path.exists(file_name_velo):
                continue

            unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
            unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)

            velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
            press = vtk_to_numpy(unstructured_grid_data_press.GetPointData().GetScalars())
            points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())
            points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())

            edges_press = get_edges(unstructured_grid_data_press, points_press, cell_size=4)
            edges_velo = get_edges(unstructured_grid_data_velo, points_velo, cell_size=8)

            sdf_velo, normal_velo = get_sdf(points_velo, points_press)
            sdf_press = np.zeros(points_press.shape[0])
            normal_press = get_normal(unstructured_grid_data_press)

            surface = {tuple(p) for p in points_press}
            exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]
            velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

            pos_ext = points_velo[exterior_indices]
            pos_surf = points_press
            sdf_ext = sdf_velo[exterior_indices]
            sdf_surf = sdf_press
            normal_ext = normal_velo[exterior_indices]
            normal_surf = normal_press
            velo_ext = velo[exterior_indices]
            velo_surf = np.array([velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3) for p in pos_surf])
            press_ext = np.zeros([len(exterior_indices), 1])
            press_surf = press

            init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
            init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
            target_ext = np.c_[velo_ext, press_ext]
            target_surf = np.c_[velo_surf, press_surf]

            surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
            pos = np.concatenate([pos_ext, pos_surf])
            init = np.concatenate([init_ext, init_surf])
            target = np.concatenate([target_ext, target_surf])
            edge_index = get_edge_index(pos, edges_press, edges_velo)

            if savedir is not None:
                save_path = os.path.join(savedir, s)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, 'x.npy'), init)
                np.save(os.path.join(save_path, 'y.npy'), target)
                np.save(os.path.join(save_path, 'pos.npy'), pos)
                np.save(os.path.join(save_path, 'surf.npy'), surf)
                np.save(os.path.join(save_path, 'edge_index.npy'), edge_index)

        surf = torch.tensor(surf)
        pos = torch.tensor(pos)
        x = torch.tensor(init)
        y = torch.tensor(target)
        edge_index = torch.tensor(edge_index)

        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0)
                mean_out = target.mean(axis=0)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0) - init.shape[0] * mean_out) / new_length
                old_length = new_length
        data = Data(pos=pos, x=x, y=y, surf=surf.bool(), edge_index=edge_index)
        dataset.append(data)

    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_in) / new_length
                std_out += (((data.y.numpy() - mean_out) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_out) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in)
        std_out = np.sqrt(std_out)

        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-8)).float()
            data.y = ((data.y - mean_out) / (std_out + 1e-8)).float()

        coef_norm = (mean_in, std_in, mean_out, std_out)
        dataset = (dataset, coef_norm)

    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)).float()
            data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)).float()

    return dataset


def get_edges(unstructured_grid_data, points, cell_size=4):
    edge_indeces = set()
    cells = vtk_to_numpy(unstructured_grid_data.GetCells().GetData()).reshape(-1, cell_size + 1)
    for i in range(len(cells)):
        for j, k in itertools.product(range(1, cell_size + 1), repeat=2):
            edge_indeces.add((cells[i][j], cells[i][k]))
            edge_indeces.add((cells[i][k], cells[i][j]))
    edges = [[], []]
    for u, v in edge_indeces:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges


def get_edge_index(pos, edges_press, edges_velo):
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    for i in range(len(edges_velo[0])):
        edges.add((indices[edges_velo[0][i]], indices[edges_velo[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index


def get_induced_graph(data, idx, num_hops):
    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx=idx, num_hops=num_hops, edge_index=data.edge_index,
                                                  relabel_nodes=True)
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def get_shape(data, max_n_point=8192, normalize=True, use_height=False):
    surf_indices = torch.where(data.surf)[0].tolist()

    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))

    shape_pc = data.pos[surf_indices].clone()

    if normalize:
        shape_pc = pc_normalize(shape_pc)

    if use_height:
        gravity_dim = 1
        height_array = shape_pc[:, gravity_dim:gravity_dim + 1] - shape_pc[:, gravity_dim:gravity_dim + 1].min()
        shape_pc = torch.cat((shape_pc, height_array), axis=1)

    return shape_pc


def create_edge_index_radius(data, r, max_neighbors=32):
    data.edge_index = nng.radius_graph(x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors)
    return data


class GraphDataset(Dataset):
    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None, coef_norm=None, valid_list=None):
        super().__init__()
        self.datalist = datalist
        self.use_height = use_height
        self.coef_norm = coef_norm
        self.valid_list = valid_list
        if not use_cfd_mesh:
            assert r is not None
            for i in range(len(self.datalist)):
                self.datalist[i] = create_edge_index_radius(self.datalist[i], r)

    def len(self):
        return len(self.datalist)

    def get(self, idx):
        data = self.datalist[idx]
        shape = get_shape(data, use_height=self.use_height)
        if self.valid_list is None:
            return self.datalist[idx].pos, self.datalist[idx].x, self.datalist[idx].y, self.datalist[idx].surf, \
                data.edge_index
        else:
            return self.datalist[idx].pos, self.datalist[idx].x, self.datalist[idx].y, self.datalist[idx].surf, \
                data.edge_index, self.valid_list[idx]

# ShapeNetCar For UPT
import scipy
import os

import numpy as np
import torch
from pathlib import Path
from kappautils.param_checking import to_3tuple, to_2tuple
from kappadata.datasets import KDDataset
from torch_geometric.nn.pool import radius, radius_graph

from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate


class ShapenetCar(KDDataset):
    # generated with torch.randperm(889, generator=torch.Generator().manual_seed(0))[:189]
    TEST_INDICES = {
        550, 592, 229, 547, 62, 464, 798, 836, 5, 732, 876, 843, 367, 496,
        142, 87, 88, 101, 303, 352, 517, 8, 462, 123, 348, 714, 384, 190,
        505, 349, 174, 805, 156, 417, 764, 788, 645, 108, 829, 227, 555, 412,
        854, 21, 55, 210, 188, 274, 646, 320, 4, 344, 525, 118, 385, 669,
        113, 387, 222, 786, 515, 407, 14, 821, 239, 773, 474, 725, 620, 401,
        546, 512, 837, 353, 537, 770, 41, 81, 664, 699, 373, 632, 411, 212,
        678, 528, 120, 644, 500, 767, 790, 16, 316, 259, 134, 531, 479, 356,
        641, 98, 294, 96, 318, 808, 663, 447, 445, 758, 656, 177, 734, 623,
        216, 189, 133, 427, 745, 72, 257, 73, 341, 584, 346, 840, 182, 333,
        218, 602, 99, 140, 809, 878, 658, 779, 65, 708, 84, 653, 542, 111,
        129, 676, 163, 203, 250, 209, 11, 508, 671, 628, 112, 317, 114, 15,
        723, 746, 765, 720, 828, 662, 665, 399, 162, 495, 135, 121, 181, 615,
        518, 749, 155, 363, 195, 551, 650, 877, 116, 38, 338, 849, 334, 109,
        580, 523, 631, 713, 607, 651, 168,
    }

    def __init__(
            self,
            split,
            data_path=None,
            radius_graph_r=None,
            radius_graph_max_num_neighbors=None,
            num_input_points_ratio=None,
            num_query_points_ratio=None,
            grid_resolution=None,
            num_supernodes=None,
            standardize_query_pos=False,
            concat_pos_to_sdf=False,
            seed=None,
            fun_dim=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split
        self.data_path = data_path
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors or int(1e10)
        self.num_supernodes = num_supernodes
        self.seed = seed
        if num_input_points_ratio is None:
            self.num_input_points_ratio = None
        else:
            self.num_input_points_ratio = to_2tuple(num_input_points_ratio)
        self.num_query_points_ratio = num_query_points_ratio
        if grid_resolution is not None:
            self.grid_resolution = to_3tuple(grid_resolution)
        else:
            self.grid_resolution = None
        self.fun_dim = fun_dim

        # define spatial min/max of simulation (for normalizing to [0, 1] and then scaling to [0, 200] for pos_embed)
        self.domain_min = torch.tensor([-2.0, -1.0, -4.5])
        self.domain_max = torch.tensor([2.0, 4.5, 6.0])
        self.scale = 200
        self.standardize_query_pos = standardize_query_pos
        self.concat_pos_to_sdf = concat_pos_to_sdf

        # mean/std for normalization (calculated on the 700 train samples)
        self.mean_in = torch.tensor([0.0])
        self.std_in = torch.tensor([1.0])
        self.mean_out = torch.tensor([-36.3099])
        self.std_out = torch.tensor([48.5743])
        self.mean_f = torch.tensor([0.0] * self.fun_dim)
        self.std_f = torch.tensor([1.0] * self.fun_dim)

        self.coef_norm = (self.mean_in, self.std_in, self.mean_out, self.std_out)

        self.source_root = Path(os.path.join(self.data_path, "preprocessed_data"))
        assert self.source_root.exists(), f"'{self.source_root.as_posix()}' doesn't exist"
        assert self.source_root.name == "preprocessed_data", f"'{self.source_root.as_posix()}' is not preprocessed folder"

        # discover uris
        self.uris = []
        for i in range(9):
            param_uri = self.source_root / f"param{i}"
            for name in sorted(os.listdir(param_uri)):
                sample_uri = param_uri / name
                if sample_uri.is_dir():
                    self.uris.append(sample_uri)
        assert len(self.uris) == 889, f"found {len(self.uris)} uris instead of 889"
        # split into train/test uris
        if split == "train":
            train_idxs = [i for i in range(len(self.uris)) if i not in self.TEST_INDICES]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
            assert len(self.uris) == 700
        elif split == "test":
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
            assert len(self.uris) == 189
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.uris)

    # noinspection PyUnusedLocal
    def getitem_pressure(self, idx, ctx=None):
        p = torch.load(self.uris[idx] / "pressure.th")
        p -= self.mean_out
        p /= self.std_out
        return p

    # noinspection PyUnusedLocal
    def getitem_grid_pos(self, idx=None, ctx=None):
        if ctx is not None and "grid_pos" in ctx:
            return ctx["grid_pos"]
        # generate positions for a regular grid (e.g. for GINO encoder)
        assert self.grid_resolution is not None
        x_linspace = torch.linspace(0, self.scale, self.grid_resolution[0])
        y_linspace = torch.linspace(0, self.scale, self.grid_resolution[1])
        z_linspace = torch.linspace(0, self.scale, self.grid_resolution[2])
        # generate positions (grid_resolution[0] * grid_resolution[1], 2)
        meshgrid = torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing="ij")
        grid_pos = torch.stack(meshgrid).flatten(start_dim=1).T
        #
        if ctx is not None:
            assert "grid_pos" not in ctx
            ctx["grid_pos"] = grid_pos
        return grid_pos

    def getitem_mesh_to_grid_edges(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.radius_graph_r is not None
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        # create graph between mesh and regular grid points
        edges = radius(
            x=mesh_pos,
            y=grid_pos,
            r=self.radius_graph_r,
            max_num_neighbors=self.radius_graph_max_num_neighbors,
        ).T
        # edges is (num_points, 2)
        return edges

    def getitem_grid_to_query_edges(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.radius_graph_r is not None
        query_pos = self.getitem_query_pos(idx, ctx=ctx)
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        # create graph between mesh and regular grid points
        edges = radius(
            x=grid_pos,
            y=query_pos,
            r=self.radius_graph_r,
            max_num_neighbors=int(1e10),
        ).T
        # edges is (num_points, 2)
        return edges

    def getitem_mesh_pos(self, idx, ctx=None):
        if ctx is not None and "mesh_pos" in ctx:
            return ctx["mesh_pos"]
        mesh_pos = self.getitem_all_pos(idx, ctx=ctx)
        # sample mesh points
        if self.num_input_points_ratio is not None:
            if self.split == "test":
                assert self.seed is not None
            if self.seed is not None:
                # deterministically downsample for evaluation
                generator = torch.Generator().manual_seed(self.seed + int(idx))
            else:
                generator = None
            # get number of samples
            if self.num_input_points_ratio[0] == self.num_input_points_ratio[1]:
                # fixed num_input_points_ratio
                end = int(len(mesh_pos) * self.num_input_points_ratio[0])
            else:
                # variable num_input_points_ratio
                lb, ub = self.num_input_points_ratio
                num_input_points_ratio = torch.rand(size=(1,), generator=generator).item() * (ub - lb) + lb
                end = int(len(mesh_pos) * num_input_points_ratio)
            # uniform sampling
            perm = torch.randperm(len(mesh_pos), generator=generator)[:end]
            mesh_pos = mesh_pos[perm]
        if ctx is not None:
            ctx["mesh_pos"] = mesh_pos
        return mesh_pos

    def getitem_all_pos(self, idx, ctx=None):
        if ctx is not None and "all_pos" in ctx:
            return ctx["all_pos"]
        all_pos = torch.load(self.uris[idx] / "mesh_points.th")
        # rescale for sincos positional embedding
        all_pos.sub_(self.domain_min).div_(self.domain_max - self.domain_min).mul_(self.scale)
        assert torch.all(0 < all_pos)
        assert torch.all(all_pos < self.scale)

        if self.fun_dim != 0:
            all_f = torch.load(self.uris[idx] / "mesh_features.th")
            all_f = (all_f - self.mean_f) / self.std_f
            all_pos = torch.cat([all_pos, all_f], dim=1)

        if ctx is not None:
            ctx["all_pos"] = all_pos
        return all_pos

    def getitem_query_pos(self, idx, ctx=None):
        if ctx is not None and "query_pos" in ctx:
            return ctx["query_pos"]
        query_pos = self.getitem_all_pos(idx, ctx=ctx)
        # sample query points
        if self.num_query_points_ratio is not None:
            if self.split == "test":
                assert self.seed is not None
            if self.seed is not None:
                # deterministically downsample for evaluation
                generator = torch.Generator().manual_seed(self.seed + int(idx))
            else:
                generator = None
            # get number of samples
            end = int(len(query_pos) * self.num_query_points_ratio)
            # uniform sampling
            perm = torch.randperm(len(query_pos), generator=generator)[:end]
            query_pos = query_pos[perm]
        # shift query_pos to [-1, 1] (required for torch.nn.functional.grid_sample)
        if self.standardize_query_pos:
            query_pos = query_pos / (self.scale / 2) - 1
        if ctx is not None:
            ctx["query_pos"] = query_pos
        return query_pos

    def _get_generator(self, idx):
        if self.split == "test":
            return torch.Generator().manual_seed(int(idx) + (self.seed or 0))
        if self.seed is not None:
            return torch.Generator().manual_seed(int(idx) + self.seed)
        return None

    # noinspection PyUnusedLocal
    def getitem_mesh_edges(self, idx, ctx=None):
        assert self.radius_graph_r is not None
        # load mesh positions
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        if self.num_supernodes is None:
            # create graph
            edges = radius_graph(
                x=mesh_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
                loop=True,
            )
        else:
            # select supernodes
            generator = self._get_generator(idx)
            perm = torch.randperm(len(mesh_pos), generator=generator)[:self.num_supernodes]
            supernodes_pos = mesh_pos[perm]
            # create edges: this can include self-loop or not depending on how many neighbors are found.
            # if too many neighbors are found, neighbors are selected randomly which can discard the self-loop
            edges = radius(
                x=mesh_pos,
                y=supernodes_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
            )
            # correct supernode index
            edges[0] = perm[edges[0]]
        return edges.T

    # noinspection PyUnusedLocal
    def getitem_sdf(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert all(self.grid_resolution[0] == grid_resolution for grid_resolution in self.grid_resolution[1:])
        sdf = torch.load(self.uris[idx] / f"sdf_res{self.grid_resolution[0]}.th")
        # check that sdf features were generated with correct positions by checking the distance to the nearest point
        if self.concat_pos_to_sdf:
            # add position to sdf (GINO uses this for interpolated FNO model)
            x_linspace = torch.linspace(-1, 1, self.grid_resolution[0])
            y_linspace = torch.linspace(-1, 1, self.grid_resolution[1])
            z_linspace = torch.linspace(-1, 1, self.grid_resolution[2])
            grid_pos = torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing="ij")
            # stack features (models expect dim_last format)
            sdf = torch.stack([sdf, *grid_pos], dim=-1)
        else:
            sdf = sdf.unsqueeze(-1)
        return sdf


    def getitem_interpolated(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.standardize_query_pos
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        # generate grid positions (these are different than getitem_gridpos because interpolate requires xy indexing)
        # it should be the same if indexing=ij since the mapping and inverse mapping consider the change in indexing
        # but for consistency with scipy.interpolate xy was chosen
        x_linspace = torch.linspace(0, self.scale, self.grid_resolution[0])
        y_linspace = torch.linspace(0, self.scale, self.grid_resolution[1])
        z_linspace = torch.linspace(0, self.scale, self.grid_resolution[2])
        grid_pos = torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing="xy")

        grid = torch.from_numpy(
            scipy.interpolate.griddata(
                mesh_pos.unbind(1),
                torch.ones_like(mesh_pos),
                grid_pos,
                method="linear",
                fill_value=0.,
            ),
        ).float()

        return grid

class ShapenetCarCollator(KDSingleCollator):
    def collate(self, batch, dataset_mode, ctx=None):
        # make sure that batch was not collated
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)

        batch, ctx = zip(*batch)
        # properties in context can have variable shapes (e.g. perm) -> delete ctx
        ctx = {}
        # dict to hold collated items
        collated_batch = {}

        # sparse mesh_pos: batch_size * (num_points, ndim) -> (batch_size * num_points, ndim)
        mesh_pos = []
        mesh_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="mesh_pos")
            mesh_lens.append(len(item))
            mesh_pos.append(item)
        collated_batch["mesh_pos"] = torch.concat(mesh_pos)

        # dense_query_pos: batch_size * (num_points, ndim) -> (batch_size, max_num_points, ndim)
        # sparse target (decoder output is converted to sparse format before loss)
        pressures = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="pressure") for sample in batch]
        # predict all positions -> pad
        query_pos = []
        query_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="query_pos")
            assert len(item) == len(pressures[i])
            query_lens.append(len(item))
            query_pos.append(item)
        collated_batch["query_pos"] = pad_sequence(query_pos, batch_first=True)
        collated_batch["pressure"] = torch.concat(pressures).unsqueeze(1)
        # create batch_idx tensor
        batch_size = len(mesh_lens)
        batch_idx = torch.empty(sum(mesh_lens), dtype=torch.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(mesh_lens)):
            end = start + mesh_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        ctx["batch_idx"] = batch_idx
        # create query_batch_idx tensor (required for test loss)
        query_batch_idx = torch.empty(sum(query_lens), dtype=torch.long)
        start = 0
        cur_query_batch_idx = 0
        for i in range(len(query_lens)):
            end = start + query_lens[i]
            query_batch_idx[start:end] = cur_query_batch_idx
            start = end
            cur_query_batch_idx += 1
        ctx["query_batch_idx"] = query_batch_idx
        # create unbatch_idx tensors (unbatch via torch_geometrics.utils.unbatch)
        # e.g. batch_size=2, num_points=[2, 3] -> unbatch_idx=[0, 0, 1, 2, 2, 2] unbatch_select=[0, 2]
        # then unbatching can be done via unbatch(dense, unbatch_idx)[unbatch_select]
        maxlen = max(query_lens)
        unbatch_idx = torch.empty(maxlen * batch_size, dtype=torch.long)
        unbatch_select = []
        unbatch_start = 0
        cur_unbatch_idx = 0
        for i in range(len(query_lens)):
            unbatch_end = unbatch_start + query_lens[i]
            unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
            unbatch_select.append(cur_unbatch_idx)
            cur_unbatch_idx += 1
            unbatch_start = unbatch_end
            padding = maxlen - query_lens[i]
            if padding > 0:
                unbatch_end = unbatch_start + padding
                unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                cur_unbatch_idx += 1
                unbatch_start = unbatch_end
        unbatch_select = torch.tensor(unbatch_select)
        ctx["unbatch_idx"] = unbatch_idx
        ctx["unbatch_select"] = unbatch_select

        # normal collation for other properties
        result = []
        for item in dataset_mode.split(" "):
            if item in collated_batch:
                result.append(collated_batch[item])
            else:
                result.append(
                    default_collate([
                        ModeWrapper.get_item(mode=dataset_mode, batch=sample, item=item)
                        for sample in batch
                    ])
                )

        return tuple(result), ctx


if __name__ == '__main__':
    import numpy as np

    file_name = '1a0bc9ab92c915167ae33d942430658c'

    root = '/data/PDE_data/mlcfd_data/training_data'
    save_path = '/data/PDE_data/mlcfd_data/preprocessed_data/param0/' + file_name
    file_name_press = 'param0/' + file_name + '/quadpress_smpl.vtk'
    file_name_velo = 'param0/' + file_name + '/hexvelo_smpl.vtk'
    file_name_press = os.path.join(root, file_name_press)
    file_name_velo = os.path.join(root, file_name_velo)
    unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
    unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)

    velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
    press = vtk_to_numpy(unstructured_grid_data_press.GetPointData().GetScalars())
    points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())
    points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())

    edges_press = get_edges(unstructured_grid_data_press, points_press, cell_size=4)
    edges_velo = get_edges(unstructured_grid_data_velo, points_velo, cell_size=8)

    sdf_velo, normal_velo = get_sdf(points_velo, points_press)
    sdf_press = np.zeros(points_press.shape[0])
    normal_press = get_normal(unstructured_grid_data_press)

    surface = {tuple(p) for p in points_press}
    exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]
    velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

    pos_ext = points_velo[exterior_indices]
    pos_surf = points_press
    sdf_ext = sdf_velo[exterior_indices]
    sdf_surf = sdf_press
    normal_ext = normal_velo[exterior_indices]
    normal_surf = normal_press
    velo_ext = velo[exterior_indices]
    velo_surf = np.array([velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3) for p in pos_surf])
    press_ext = np.zeros([len(exterior_indices), 1])
    press_surf = press

    init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
    init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
    target_ext = np.c_[velo_ext, press_ext]
    target_surf = np.c_[velo_surf, press_surf]

    surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
    pos = np.concatenate([pos_ext, pos_surf])
    init = np.concatenate([init_ext, init_surf])
    target = np.concatenate([target_ext, target_surf])

    edge_index = get_edge_index(pos, edges_press, edges_velo)

    data = Data(pos=torch.tensor(pos), edge_index=torch.tensor(edge_index))
    data = create_edge_index_radius(data, r=0.2)
    x, y = data.edge_index
    import torch_geometric

    print(max(torch_geometric.utils.degree(x)), max(torch_geometric.utils.degree(y)))

    print(points_velo.shape, points_press.shape)
    print(surf.shape, pos.shape, init.shape, target.shape, edge_index.shape)
