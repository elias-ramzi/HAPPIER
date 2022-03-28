from .percentage import percentage


def format_for_latex(metrics):
    try:
        print_for_latex = []
        print_for_latex.append(str(percentage(metrics["P@1_level0"])))
        stop = False
        level = 0
        while not stop:
            try:
                print_for_latex.append(str(percentage(metrics[f"AP_level{level}"])))
                level += 1
            except KeyError:
                stop = True

        for mlt in ["H-AP_multi", "ASI_multi", "NDCG_multi"]:
            try:
                print_for_latex.append(str(percentage(metrics[mlt])))
            except KeyError:
                print_for_latex.append("-")

        return " & ".join(print_for_latex)
    except Exception as exc:
        print("Could not format for latex")
        print(exc)
        return None


def format_for_latex_dyml(metrics):
    try:
        print_for_latex = []
        print_for_latex.append(str(percentage(metrics.get("AP_level0", "-"))))
        print_for_latex.append(str(percentage(metrics.get("ASI_level0", "-"))))
        print_for_latex.append(str(percentage(metrics.get("P@1_level0", "-"))))
        return " & ".join(print_for_latex)
    except Exception as exc:
        print("Could not format for latex")
        print(exc)
        return None
