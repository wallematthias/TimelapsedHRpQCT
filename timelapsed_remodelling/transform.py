import os
import yaml
import networkx as nx
import SimpleITK as sitk
from glob import glob
from . import custom_logger

class TimelapsedTransformation:
    def __init__(self):
        """
        Initialize the TimelapsedTransformation class.

        This class is designed to store a graph of transformations and metrics,
        where each node represents a transformation and each edge represents
        a directed transformation from one node to another.

        The class provides methods to add transformations, check if a transformation
        exists between two nodes, get the shortest path of transformations between
        two nodes, and perform resampling using the computed transformations.

        Attributes:
            data (networkx.DiGraph): The graph that stores transformations and metrics.
        """
        self.data = nx.DiGraph()

    def add_transform(self, transform, source=0, target=0, metric=0):
        """
        Add a transformation to the graph.

        Args:
            transform (SimpleITK.Transform): The transformation to add.
            source (int): The source node index.
            target (int): The target node index.
            metric (float): The final metric associated with the transformation.

        Returns:
            None
        """
        # Adding transform
        self.data.add_edge(source, target, transform=transform, finalMetric=metric, label='{}TO{}'.format(source, target))
        # Adding inverse transform
        self.data.add_edge(target, source, transform=transform.GetInverse(), finalMetric=metric, label='{}TO{}'.format(source, target))
        # Adding source eigentransform if not exist
        if not self.data.has_edge(source, source):
            self.data.add_edge(source, source, transform=sitk.Euler3DTransform(), finalMetric=1, label='{}TO{}'.format(source, source))
        # Adding target eigentransform if not exist
        if not self.data.has_edge(target, target):
            self.data.add_edge(target, target, transform=sitk.Euler3DTransform(), finalMetric=1, label='{}TO{}'.format(target, target))

    def exists(self, source=0, target=0):
        """
        Check if a transformation exists between two nodes in the graph.

        Args:
            source (int): The source node index.
            target (int): The target node index.

        Returns:
            int: 1 if the transformation exists, 0 otherwise.
        """
        try:
            # detect shortest path for requested transform
            sp = nx.shortest_path(self.data, source=source, target=target)
            return 1
        except nx.NetworkXNoPath:
            return 0

    def get_transform(self, source=0, target=0):
        """
        Get the composite transformation and associated metrics for the shortest path
        between two nodes in the graph.

        Args:
            source (int): The source node index.
            target (int): The target node index.

        Returns:
            tuple: A tuple containing the composite transform, a list of metrics for
                   each transformation in the path, and the shortest path as a list of nodes.
        """
        if self.exists(source, target):
            # Get shortest path as a combination of transformations
            sp = nx.shortest_path(self.data, source=source, target=target)
            path_graph = nx.path_graph(sp)

            # Extract the edge metrics again
            transformations = [self.data.edges[ea[0], ea[1]]['transform'] for ea in path_graph.edges()]
            metrics = [self.data.edges[ea[0], ea[1]]['finalMetric'] for ea in path_graph.edges()]
            labels = [self.data.edges[ea[0], ea[1]]['label'] for ea in path_graph.edges()]
            custom_logger.info("Shortest transformation path: {} with metrics: {}: ".format(sp, metrics))

            # Making the composite transform
            composite_transform = sitk.CompositeTransform(transformations)
            composite_transform.FlattenTransform()

            return composite_transform #, metrics, list(sp)
        else:
            # Just return something in case
            custom_logger.info('Warning: Returning empty transformation')
            return sitk.Euler3DTransform()

    def transform(self, image, source, target):
        """
        Perform resampling of the input image using the computed transformation.

        Args:
            image (numpy.ndarray): The input image to be resampled.
            source (int): The source node index.
            target (int): The target node index.
            interpolator (str): The type of interpolator for resampling.

        Returns:
            numpy.ndarray: The resampled image as a NumPy array.
        """
        # Get transformation
        transform = self.get_transform(source=source, target=target)

        # Cast ITK images
        im = sitk.Cast(sitk.GetImageFromArray(image.astype(int)), sitk.sitkFloat32)

        # Resample Images and get array
        resampled_image = sitk.GetArrayFromImage(sitk.Resample(im, im, transform, sitk.sitkLinear, 0.0, im.GetPixelID()))

        return resampled_image

    def save_transform(self, path):
        """
        Save the transformations and associated metrics to disk.

        Args:
            path (str): The directory path to save the transformations.

        Returns:
            None
        """
        data = nx.to_dict_of_dicts(self.data)

        if not os.path.exists(path):
            os.makedirs(path)

        for key1, lvl2 in data.items():
            for key2, lvl3 in lvl2.items():
                if key1 < key2:
                    filename = '{}_{}_{}.tfm'.format(key1, key2, os.path.basename(path))
                    transform = data[key1][key2]['transform']
                    transform = self._simplify_composite_transform(transform)
                    sitk.WriteTransform(transform, os.path.join(path, filename))

                    with open(os.path.join(path, filename.replace('.tfm', '.yml')), 'w') as handle:
                        yaml.dump({'metric': float(data[key1][key2]['finalMetric'])}, handle, default_flow_style=True)

    def load_transform(self, path):
        """
        Load the transformations and associated metrics from disk.

        Args:
            path (str): The directory path containing the transformations.

        Returns:
            None
        """
        #custom_logger.info(os.path.join(path, '*{}.tfm'.format(os.path.basename(path))))
        #tpaths = glob(os.path.join(path, '*{}.tfm'.format(os.path.basename(path))))
        
        #for tpath in tpaths:
        tpath = path
        sitktransform = sitk.CompositeTransform(sitk.ReadTransform(tpath))
        
        with open(tpath.replace('.tfm', '.yml'), "r") as stream:
            metric = yaml.load(stream, Loader=yaml.FullLoader)['metric']
        source, target = os.path.basename(tpath).split('_')[:2]
        custom_logger.info('Adding transform from {} to {} with metric={}'.format(source, target, format(float(metric), '.4f')))

        self.add_transform(sitktransform, source=source, target=target, metric=float(metric))

    @staticmethod
    def _simplify_composite_transform(composite_transform):
        """
        Simplify the composite transform by removing empty transformations.

        Args:
            composite_transform (SimpleITK.CompositeTransform): The composite transform.

        Returns:
            SimpleITK.CompositeTransform: The simplified composite transform.
        """
        composite_transform = sitk.CompositeTransform(composite_transform)
        composite_transform.FlattenTransform()
        num_of_transforms = composite_transform.GetNumberOfTransforms()
        non_empty = [composite_transform.GetNthTransform(n) for n in range(num_of_transforms) if
                     composite_transform.GetNthTransform(n).GetParameters() != composite_transform.GetNthTransform(n).GetInverse().GetParameters()]

        if len(non_empty) > 0:
            composite_transform = sitk.CompositeTransform(non_empty)
        else:
            composite_transform = sitk.CompositeTransform([sitk.Euler3DTransform()])

        composite_transform.FlattenTransform()
        return composite_transform
