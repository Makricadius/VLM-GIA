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
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.ravel()
    axs[0].plot(alphas_arr, cl_arr, marker='o')
    axs[0].set_title('CL (total)')
    axs[0].set_xlabel('alpha')

    axs[1].plot(alphas_arr, cm0y_arr, marker='o')
    axs[1].set_title('cm0y')
    axs[1].set_xlabel('alpha')

    axs[2].plot(alphas_arr, cma_arr, marker='o')
    axs[2].set_title('cma')
    axs[2].set_xlabel('alpha')

    axs[3].plot(alphas_arr, xcp_arr, marker='o')
    axs[3].set_title('xcp')
    axs[3].set_xlabel('alpha')

    axs[4].plot(alphas_arr, cd_arr, marker='o')
    axs[4].set_title('CD (total)')
    axs[4].set_xlabel('alpha')

    axs[5].axis('off')
    fig.tight_layout()

    # Prepare spanwise plots: choose a few alphas to overlay
    if spanwise_alphas is None:
        # pick up to 5 alphas evenly spaced
        n = min(5, len(alphas))
        idx = np.linspace(0, len(alphas) - 1, n, dtype=int)
        spanwise_alphas = [alphas[i] for i in idx]

    # CL spanwise
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for a in spanwise_alphas:
        clj = span_data[a].get('clj')
        if clj is None:
            continue
        x = np.arange(len(clj))
        ax2.plot(x, clj, marker='o', label=f'alpha={a}')
    ax2.set_title('Spanwise sectional CL')
    ax2.set_xlabel('span station index')
    ax2.set_ylabel('cl_j')
    ax2.legend()
    fig2.tight_layout()

    # CM spanwise
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for a in spanwise_alphas:
        cmj = span_data[a].get('cm0y_j')
        if cmj is None:
            continue
        x = np.arange(len(cmj))
        ax3.plot(x, cmj, marker='o', label=f'alpha={a}')
    ax3.set_title('Spanwise sectional cm0y_j')
    ax3.set_xlabel('span station index')
    ax3.set_ylabel('cm0y_j')
    ax3.legend()
    fig3.tight_layout()

    # CD spanwise and induced velocity
    fig4, ax4 = plt.subplots(2, 1, figsize=(8, 8))
    for a in spanwise_alphas:
        cdj = span_data[a].get('cdj')
        w = span_data[a].get('w_ind')
        if cdj is not None:
            x = np.arange(len(cdj))
            ax4[0].plot(x, cdj, marker='o', label=f'alpha={a}')
        if w is not None:
            x = np.arange(len(w))
            ax4[1].plot(x, w, marker='o', label=f'alpha={a}')
    ax4[0].set_title('Spanwise sectional CD_j')
    ax4[0].set_xlabel('span station index')
    ax4[0].legend()
    ax4[1].set_title('Induced velocity (w_induced)')
    ax4[1].set_xlabel('span station index')
    ax4[1].legend()
    fig4.tight_layout()

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
