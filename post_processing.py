import numpy as np
import matplotlib.pyplot as plt

def plot_aero_characteristics(aero_wing, spanwise_alphas=None, show=True):
    """Plot aerodynamic results stored in an Aerdynamic_wing object's `alpha_memory`.

    Parameters
    - aero_wing: instance of Aerdynamic_wing with `alpha_memory` populated.
    - alphas: optional list of alpha values to include (defaults to all stored, sorted).
    - spanwise_alphas: optional subset of alphas to overlay on spanwise plots
      (defaults to up to 5 evenly spaced alphas from the selected set).
    - show: whether to call `plt.show()` (useful for testing).
    """
    mem = getattr(aero_wing, "alpha_memory", None)
    if not mem:
        raise ValueError("aero_wing.alpha_memory is empty or missing")

    all_alphas = sorted(mem.keys())
    # always use the alphas present in alpha_memory
    alphas = all_alphas
    if not alphas:
        raise ValueError("no alphas stored in aero_wing.alpha_memory")


    cl_list = []
    cm0y_list = []
    cma_list = []
    xcp_list = []
    cd_list = []

    # collect spanwise arrays for later plotting
    span_data = {}

    for a in alphas:
        A = mem[a]["A"]
        # Safely extract values (A format: {"cl":[cl,clj,clij],"cm":[cm0y,cm0y_j,cm0y_ij,cma,cmc4],"xcp":xcp,"cd":[cd,cdj,w_inducido]})
        cl = np.asarray(A.get("cl", [np.nan])[0])
        cl_list.append(float(cl))

        cm0y = np.asarray(A.get("cm", [np.nan])[0])
        cm0y_list.append(float(cm0y))

        cma = np.asarray(A.get("cm", [np.nan, np.nan, np.nan, np.nan])[3])
        cma_list.append(float(cma))

        xcp = A.get("xcp", np.nan)
        xcp_list.append(float(xcp))

        cd = np.asarray(A.get("cd", [np.nan])[0])
        cd_list.append(float(cd))

        # store spanwise arrays if present
        span_data[a] = {
            "clj": np.asarray(A.get("cl", [None, None, None])[1]) if len(A.get("cl", [])) > 1 else None,
            "cm0y_j": np.asarray(A.get("cm", [None, None])[1]) if len(A.get("cm", [])) > 1 else None,
            "cdj": np.asarray(A.get("cd", [None, None])[1]) if len(A.get("cd", [])) > 1 else None,
            "w_ind": np.asarray(A.get("cd", [None, None, None])[2]) if len(A.get("cd", [])) > 2 else None,
        }

    # Convert lists to arrays
    alphas_arr = np.array(alphas)
    cl_arr = np.array(cl_list)
    cm0y_arr = np.array(cm0y_list)
    cma_arr = np.array(cma_list)
    xcp_arr = np.array(xcp_list)
    cd_arr = np.array(cd_list)

    # Scalar plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs = axs.ravel()
    # Identify integer alpha entries for marker overlay
    int_mask = np.isclose(alphas_arr, np.round(alphas_arr))

    # CL: continuous line plus markers only at integer alphas
    ln = axs[0].plot(alphas_arr, cl_arr, linestyle='-')[0]
    if int_mask.any():
        axs[0].plot(alphas_arr[int_mask], cl_arr[int_mask], marker='o', linestyle='None', color=ln.get_color(),
                    markerfacecolor='none', markersize=4)
    axs[0].set_title('CL (total)')
    axs[0].set_xlabel('alpha')

    # cm0y and cma on same axes: lines with integer markers
    l1 = axs[1].plot(alphas_arr, cm0y_arr, linestyle='-', label='cm0y')[0]
    l2 = axs[1].plot(alphas_arr, cma_arr, linestyle='-', label='cma')[0]
    if int_mask.any():
        axs[1].plot(alphas_arr[int_mask], cm0y_arr[int_mask], marker='o', linestyle='None', color=l1.get_color(),
                    markerfacecolor='none', markersize=4)
        axs[1].plot(alphas_arr[int_mask], cma_arr[int_mask], marker='o', linestyle='None', color=l2.get_color(),
                    markerfacecolor='none', markersize=4)
    axs[1].set_title('cm0y and cma')
    axs[1].set_xlabel('alpha')
    axs[1].legend()

    # xcp: continuous line with integer markers
    ln_xcp = axs[2].plot(alphas_arr, xcp_arr, linestyle='-')[0]
    if int_mask.any():
        axs[2].plot(alphas_arr[int_mask], xcp_arr[int_mask], marker='o', linestyle='None', color=ln_xcp.get_color(),
                    markerfacecolor='none', markersize=4)
    axs[2].set_title('xcp')
    axs[2].set_xlabel('alpha')

    # CD: continuous line with integer markers
    ln_cd = axs[3].plot(alphas_arr, cd_arr, linestyle='-')[0]
    if int_mask.any():
        axs[3].plot(alphas_arr[int_mask], cd_arr[int_mask], marker='o', linestyle='None', color=ln_cd.get_color(),
                    markerfacecolor='none', markersize=4)
    axs[3].set_title('CD (total)')
    axs[3].set_xlabel('alpha')

    fig.tight_layout()

    # Prepare spanwise plots: choose a few alphas to overlay
    if spanwise_alphas is None:
        # Prefer integer alpha values present in alpha_memory (e.g. -5, -4, ...).
        # This selects every integer angle that exists; if none are integer,
        # fall back to selecting up to 5 evenly spaced alphas.
        int_alphas = [a for a in alphas if float(a).is_integer()]
        if int_alphas:
            spanwise_alphas = int_alphas
        else:
            n = min(5, len(alphas))
            idx = np.linspace(0, len(alphas) - 1, n, dtype=int)
            spanwise_alphas = [alphas[i] for i in idx]

    # Extract panel spanwise positions (y-coordinates) from singularity model
    # Use control point y-coordinates as the spanwise position for each panel
    if hasattr(aero_wing, 'model') and hasattr(aero_wing.model, 'yc'):
        y_positions = np.asarray(aero_wing.model.yc)
        # Convert to percentage of semi-span (keep sign to show both wings)
        if hasattr(aero_wing, 'wing') and hasattr(aero_wing.wing, 'b'):
            semi_span = aero_wing.wing.b / 2
            y_positions = (y_positions / semi_span) * 100
            xlabel = '% of semi-span'
        else:
            xlabel = 'Spanwise position (y)'
    else:
        # Fallback to index if y-coordinates not available
        y_positions = None
        xlabel = 'span station index'

    # Combined spanwise plots: CL, CM, CD_j and induced velocity in one 2x2 figure
    fig_s, axs_s = plt.subplots(2, 2, figsize=(16, 10))
    axs_s = axs_s.ravel()
    for a in spanwise_alphas:
        clj = span_data[a].get('clj')
        cmj = span_data[a].get('cm0y_j')
        cdj = span_data[a].get('cdj')
        w = span_data[a].get('w_ind')

        if clj is not None:
            x = y_positions if y_positions is not None else np.arange(len(clj))
            axs_s[0].plot(x, clj, marker='o', linestyle='-', markerfacecolor='none', markersize=4, label=f'alpha={a}')
        if cmj is not None:
            x = y_positions if y_positions is not None else np.arange(len(cmj))
            axs_s[1].plot(x, cmj, marker='o', linestyle='-', markerfacecolor='none', markersize=4, label=f'alpha={a}')
        if cdj is not None:
            x = y_positions if y_positions is not None else np.arange(len(cdj))
            axs_s[2].plot(x, cdj, marker='o', linestyle='-', markerfacecolor='none', markersize=4, label=f'alpha={a}')
        if w is not None:
            x = y_positions if y_positions is not None else np.arange(len(w))
            axs_s[3].plot(x, w, marker='o', linestyle='-', markerfacecolor='none', markersize=4, label=f'alpha={a}')

    axs_s[0].set_title('Spanwise sectional CL')
    axs_s[0].set_xlabel(xlabel)
    axs_s[0].set_ylabel('cl_j')
    axs_s[0].grid(True, alpha=0.3)

    axs_s[1].set_title('Spanwise sectional cm0y_j')
    axs_s[1].set_xlabel(xlabel)
    axs_s[1].set_ylabel('cm0y_j')
    axs_s[1].grid(True, alpha=0.3)

    axs_s[2].set_title('Spanwise sectional CD_j')
    axs_s[2].set_xlabel(xlabel)
    axs_s[2].grid(True, alpha=0.3)

    axs_s[3].set_title('Induced velocity (w_induced)')
    axs_s[3].set_xlabel(xlabel)
    axs_s[3].grid(True, alpha=0.3)

    # Adjust layout first to make room for legend
    fig_s.tight_layout(rect=[0, 0, 0.88, 1])
    
    # Create a single legend to the right of all subplots
    handles, labels = axs_s[0].get_legend_handles_labels()
    fig_s.legend(handles, labels, loc='center left', bbox_to_anchor=(0.88, 0.5), frameon=True)

    if show:
        plt.show()

    return {
        'alphas': alphas_arr,
        'cl': cl_arr,
        'cm0y': cm0y_arr,
        'cma': cma_arr,
        'xcp': xcp_arr,
        'cd': cd_arr,
        'span_data': span_data,
    }
