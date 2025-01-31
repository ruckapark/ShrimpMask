from tensorflow.keras import backend as K

def weighted_binary_crossentropy(y_true, y_pred, weight=10):
    """
    Custom loss function that penalizes more heavily the white pixels.

    Parameters:
    y_true: Ground truth labels.
    y_pred: Predicted labels.
    weight: Weight for white pixels. Black pixels will have a weight of 1.

    Returns:
    Loss value.
    """
    # Calculate the binary cross-entropy loss
    bce = K.binary_crossentropy(y_true, y_pred)

    # Create a weight mask: weight for white pixels, 1 for black pixels
    weight_mask = y_true * (weight - 1) + 1

    # Apply the weight mask to the loss
    weighted_bce = bce * weight_mask

    # Return the mean loss
    return K.mean(weighted_bce)