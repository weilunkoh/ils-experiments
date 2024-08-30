def handle_zero(truth_class: str, pred_proba: list, class_names: list):
    truth_class_idx = class_names.index(truth_class)
    if pred_proba[truth_class_idx] <= 0:
        pred_proba[truth_class_idx] = -1
    return pred_proba
