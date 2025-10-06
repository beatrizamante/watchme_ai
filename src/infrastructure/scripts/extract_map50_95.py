def _extract_map50_95(results, model=None, data=None):
    try:
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95", "mAP50-95", "map50-95"):
                if k in results.results_dict:
                    return float(results.results_dict[k])
    except (KeyError, TypeError, ValueError):
        pass

    try:
        if hasattr(results, "metrics") and isinstance(results.metrics, dict):
            for k in ("map50_95", "mAP50-95", "map"):
                if k in results.metrics:
                    return float(results.metrics[k])
    except (KeyError, TypeError, ValueError):
        pass

    try:
        if model is not None and data is not None:
            val_res = model.val(data=data)
            if hasattr(val_res, "box") and hasattr(val_res.box, "map"):
                return float(val_res.box.map)
    except (KeyError, TypeError, ValueError):
        pass

    return 0.0