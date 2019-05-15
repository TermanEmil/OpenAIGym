from list_tools import flatten


def flatten_layer_weights(layer):
    w = flatten(layer.get_weights()[0])
    b = layer.get_weights()[1]
    return w + list(b)


def flatten_all_layers_weights(layers):
    return flatten((flatten_layer_weights(layer) for layer in layers))


def extract_flat_weights_from_keras_model(keras_model):
    return flatten_all_layers_weights(keras_model.layers)


def extract_nb_of_weights_per_layer(model):
    weights_per_layer = []
    for layer in model.layers:
        nb_of_weights = len(flatten_layer_weights(layer))
        weights_per_layer.append(nb_of_weights)

    return weights_per_layer


def apply_weights_to_layer(layer, flat_weights):
    weights = layer.get_weights()
    w_i = 0

    # Non bias weights
    non_bias_w = weights[0]
    for i in range(len(non_bias_w)):
        for j in range(len(non_bias_w[i])):
            non_bias_w[i][j] = flat_weights[w_i]
            w_i += 1

    # Bias weights
    bias_w = weights[1]
    for i in range(len(bias_w)):
        bias_w[i] = flat_weights[w_i]
        w_i += 1

    layer.set_weights(weights)
    assert w_i == len(flat_weights)


def apply_weights_to_keras_model(all_weights, model, nb_of_weights_per_layer):
    weights = all_weights

    for layer_i in range(len(model.layers)):
        layer = model.layers[layer_i]
        nb_of_weights_on_this_layer = nb_of_weights_per_layer[layer_i]
        flat_weights_for_this_layer = weights[:nb_of_weights_on_this_layer]

        apply_weights_to_layer(layer, flat_weights_for_this_layer)

        weights = weights[nb_of_weights_on_this_layer:]

    assert len(weights) == 0
