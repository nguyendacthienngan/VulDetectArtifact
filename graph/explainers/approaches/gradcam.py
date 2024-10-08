import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Callable, List, Tuple, Union

from graph.explainers.approaches.common import WalkBase


import captum.attr._core.layer.grad_cam as ca
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_inputs,
)
from captum._utils.gradient import apply_gradient_requirements, undo_gradient_requirements, compute_layer_gradients_and_eval
from captum._utils.typing import (
    TargetType,
)

EPS = 1e-15

def normalize(x: torch.Tensor):
    x -= x.min()
    if x.max() == 0:
        return torch.zeros(x.size(), device=x.device)
    x /= x.max()
    return x

class GradCAM(WalkBase):
    def __init__(self, model, epochs=100, lr=0.01, explain_graph=False, molecule=False, device="cpu"):
        super().__init__(model, epochs, lr, explain_graph, molecule, device)
        self.layer = model.key_layer

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs):
        """
        Given a sample, this function will return its predicted masks and corresponding predictions
        for evaluation
        :param x: Tensor - Hiden features of all vertexes  [num_nodes, node_features]
        :param edge_index: Tensor - All connected edge between vertexes/nodes   [2, edge_nums]
        :param kwargs:
        :return:
        """
        self.model.eval()
        super().forward(x, edge_index)
        model = self.model
        self.explain_method = GraphLayerGradCam(model, self.layer, device=self.device)
        # self.explain_method = GraphLayerGradCam(model, model.GGNN)
        # --- setting end ---

        ex_label = 1
        attr_wo_relu = self.explain_method.attribute(x, ex_label, additional_forward_args=edge_index)   # [num_node, 1]
        mask = normalize(attr_wo_relu.relu())
        mask = mask.squeeze()
        # mask = (mask[self_loop_edge_index[0]] + mask[self_loop_edge_index[1]]) / 2
        # mask = self.control_sparsity(mask, kwargs.get('sparsity'))
        # Store related predictions for further evaluation.

        # with torch.no_grad():
        #     with self.connect_mask(self):
        #         related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return mask.detach()

    def explain(self, x, edge_index):
        mask = self.forward(x, edge_index)
        sorted_results = mask.sort(descending=True).indices.cpu()
        return sorted_results, True


class GraphLayerGradCam(ca.LayerGradCam):

    def __init__(
            self,
            forward_func: Callable,  # GNN model
            layer: nn.Module,
            device_ids: Union[None, List[int]] = None,
            device = "cpu",
    ) -> None:
        super().__init__(forward_func, layer, device_ids)
        self.device = device

    def attribute(
            self,
            inputs: Union[Tensor, Tuple[Tensor, ...]],   # [num_nodes, num_features]
            target: TargetType = None,  # [1]
            additional_forward_args: Any = None,   # [2, num_edges]
            attribute_to_layer_input: bool = False,
            relu_attributions: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which attributions
                        are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to the
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the outputs of internal layers, depending on whether we
                        attribute to the input or output, are single tensors.
                        Support for multiple tensors will be added later.
                        Default: False
            relu_attributions (bool, optional): Indicates whether to
                        apply a ReLU operation on the final attribution,
                        returning only non-negative attributions. Setting this
                        flag to True matches the original GradCAM algorithm,
                        otherwise, by default, both positive and negative
                        attributions are returned.
                        Default: False

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Attributions based on GradCAM method.
                        Attributions will be the same size as the
                        output of the given layer, except for dimension 2,
                        which will be 1 due to summing over channels.
                        Attributions are returned in a tuple based on whether
                        the layer inputs / outputs are contained in a tuple
                        from a forward hook. For standard modules, inputs of
                        a single tensor are usually wrapped in a tuple, while
                        outputs of a single tensor are not.
        Examples::

            # >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            # >>> # and returns an Nx10 tensor of class probabilities.
            # >>> # It contains a layer conv4, which is an instance of nn.conv2d,
            # >>> # and the output of this layer has dimensions Nx50x8x8.
            # >>> # It is the last convolution layer, which is the recommended
            # >>> # use case for GradCAM.
            # >>> net = ImageClassifier()
            # >>> layer_gc = LayerGradCam(net, net.conv4)
            # >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            # >>> # Computes layer GradCAM for class 3.
            # >>> # attribution size matches layer output except for dimension
            # >>> # 1, so dimensions of attr would be Nx1x8x8.
            # >>> attr = layer_gc.attribute(input, 3)
            # >>> # GradCAM attributions are often upsampled and viewed as a
            # >>> # mask to the input, since the convolutional layer output
            # >>> # spatially matches the original input image.
            # >>> # This can be done with LayerAttribution's interpolate method.
            # >>> upsampled_attr = LayerAttribution.interpolate(attr, (32, 32))
        """
        inputs = _format_inputs(inputs)
        additional_forward_args = _format_additional_forward_args(  # tuple(additional_forward_args)
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals, is_layer_tuple = compute_layer_gradients_and_eval(
            self.forward_func,  # model
            self.layer,  # ggnn layer
            inputs=inputs,  # x
            target_ind=torch.LongTensor([target]).to(self.device), # label
            additional_forward_args=additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        undo_gradient_requirements(inputs, gradient_mask)

        # Gradient Calculation end

        # what I add: shape from PyG to General PyTorch
        layer_gradients = tuple(layer_grad.transpose(0, 1).unsqueeze(0)
                                for layer_grad in layer_gradients)

        layer_evals = tuple(layer_eval.transpose(0, 1).unsqueeze(0)
                            for layer_eval in layer_evals)
        # end

        summed_grads = tuple(
            torch.mean(
                layer_grad,
                dim=tuple(x for x in range(2, len(layer_grad.shape))),
                keepdim=True,
            )
            for layer_grad in layer_gradients
        )

        scaled_acts = tuple(
            torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
            for summed_grad, layer_eval in zip(summed_grads, layer_evals)
        )
        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)

        # what I add: shape from General PyTorch to PyG

        scaled_acts = tuple(scaled_act.squeeze(0).transpose(0, 1)
                            for scaled_act in scaled_acts)

        # end

        return _format_output(is_layer_tuple, scaled_acts)