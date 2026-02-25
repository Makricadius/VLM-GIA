import numpy as np
import matplotlib.pyplot as plt

def plot_aero_characteristics(aero_wing, plotted_alphas=None, show=True):
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

    # Calculate any missing alphas from plotted_alphas if provided
    if plotted_alphas is not None:
        for alpha in plotted_alphas:
            if alpha not in mem:
                print(f"Calculating alpha={alpha}°...")
                aero_wing.calculate(alpha)

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

    # Extract panel spanwise positions (y-coordinates) from singularity model
    # Reduce chordwise panels to a single spanwise vector (length Nb)
    if hasattr(aero_wing, 'model') and hasattr(aero_wing.model, 'yc'):
        y_positions = np.asarray(aero_wing.model.yc)
        nb = getattr(aero_wing.model, 'Nb', None)
        nc = getattr(aero_wing.model, 'Nc', None)
        if nb is not None and nc is not None and y_positions.size == nb * nc:
            y_positions = y_positions.reshape((nc, nb)).mean(axis=0)
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

    # Scalar plots (2x3 grid to accommodate CD vs CL polar plot)
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    axs = axs.ravel()
    # Identify integer alpha entries for marker overlay
    int_mask = np.isclose(alphas_arr, np.round(alphas_arr))

    # Decompose sectional CL into basic Cl0 and additional Cla using linear system
    # Cl(y) = Cl0(y) + Cla(y) * CL
    # Using two different CL values to solve for Cl0 and Cla
    axs[0].set_title(r'Spanwise $C_l(y) = C_{l0}(y) + C_{la}(y) \cdot C_L$')
    axs[0].set_xlabel('Spanwise position')
    axs[0].set_ylabel('Sectional CL')
    axs[0].grid(True, alpha=0.3)
    
    # Select two alphas for decomposition (prefer well-separated values)
    if len(alphas) >= 2:
        # Use first and last alpha for maximum separation
        idx_a1 = 0
        idx_a2 = len(alphas) - 1
        alpha1 = alphas[idx_a1]
        alpha2 = alphas[idx_a2]
        
        # Get sectional CL(y) for both alphas
        clj_1 = span_data[alpha1].get('clj')
        clj_2 = span_data[alpha2].get('clj')
        
        # Get total CL for both alphas
        cl_1 = cl_arr[idx_a1]
        cl_2 = cl_arr[idx_a2]
        
        if clj_1 is not None and clj_2 is not None and cl_1 != cl_2:
            # Solve linear system:
            # clj_1 = cl0 + cla * cl_1
            # clj_2 = cl0 + cla * cl_2
            # Solution:
            # cla = (clj_1 - clj_2) / (cl_1 - cl_2)
            # cl0 = clj_1 - cla * cl_1
            
            cla = (clj_1 - clj_2) / (cl_1 - cl_2)
            cl0 = clj_1 - cla * cl_1
            
            # Plot decomposition
            if y_positions is not None:
                x = y_positions
            else:
                x = np.arange(len(cl0))
            
            # Plot basic CL
            axs[0].plot(x, cl0, marker='o', linestyle='-', markerfacecolor='none', 
                        markersize=4, linewidth=2)
            
            # Plot additional CL gradient
            axs[0].plot(x, cla, marker='s', linestyle='--', markerfacecolor='none', 
                        markersize=4, linewidth=2)
            
            # Add annotation showing which alphas were used
            axs[0].text(0.5, 0.02, f'Decomposed from:\nα₁={alpha1:.2f}°, C_L₁={cl_1:.4f}\nα₂={alpha2:.2f}°, C_L₂={cl_2:.4f}',
                       transform=axs[0].transAxes,
                       verticalalignment='bottom', horizontalalignment='center',
                       fontsize=9, color='black', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[0].text(0.5, 0.5, 'Insufficient data for decomposition\n(need at least 2 alpha values)',
                   transform=axs[0].transAxes,
                   verticalalignment='center', horizontalalignment='center',
                   fontsize=10)

    # CL: markers only at integer alphas and linear interpolation fit line
    if int_mask.any():
        axs[1].plot(alphas_arr[int_mask], cl_arr[int_mask], marker='o', linestyle='None', color='blue', markerfacecolor='none', markersize=4)
    # Mark plotted_alphas with red dots if provided
    if plotted_alphas is not None:
        plotted_mask = np.isin(alphas_arr, plotted_alphas)
        if plotted_mask.any():
            axs[1].plot(alphas_arr[plotted_mask], cl_arr[plotted_mask], marker='o', linestyle='None', color='red', markerfacecolor='red', markersize=5)
    axs[1].set_title(r'$C_L$')
    axs[1].set_xlabel('Angle of attack [º]')
    axs[1].grid(True, alpha=0.3)
    # Annotate fitted line inclination next to the CL line
    if len(alphas_arr) > 1:
        cl_slope, cl_intercept = np.polyfit(alphas_arr, cl_arr, 1)
        cl_slope_rad = cl_slope * 180 / np.pi
        alpha_fit = np.linspace(alphas_arr.min(), alphas_arr.max(), 200)
        cl_fit = cl_intercept + cl_slope * alpha_fit
        ln = axs[1].plot(alpha_fit, cl_fit, linestyle='--', linewidth=1.5, color='blue')[0]
        axs[1].text(0.98, 0.02, rf'$C_{{L0}}$ = {cl_intercept:.4f}' + '\n' + rf'$C_{{L\alpha}}$ = {cl_slope_rad:.4f} 1/rad',
                    transform=axs[1].transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    color=ln.get_color(), fontsize=9)

    # CDi: markers only at integer alphas and quadratic interpolation fit line
    if int_mask.any():
        axs[2].plot(alphas_arr[int_mask], cd_arr[int_mask], marker='o', linestyle='None', color='blue', markerfacecolor='none', markersize=4)
    # Mark plotted_alphas with red dots if provided
    if plotted_alphas is not None:
        plotted_mask = np.isin(alphas_arr, plotted_alphas)
        if plotted_mask.any():
            axs[2].plot(alphas_arr[plotted_mask], cd_arr[plotted_mask], marker='o', linestyle='None', color='red', markerfacecolor='red', markersize=5)
    fit_color = None
    if len(alphas_arr) >= 3:
        k_cd, j_cd, a_cd = np.polyfit(alphas_arr, cd_arr, 2)
        alpha_fit = np.linspace(alphas_arr.min(), alphas_arr.max(), 200)
        cd_fit = a_cd + j_cd * alpha_fit + k_cd * alpha_fit**2
        fit_color = axs[2].plot(alpha_fit, cd_fit, linestyle='--', linewidth=1.5, color='blue')[0].get_color()
    else:
        k_cd, j_cd, a_cd = np.nan, np.nan, np.nan

    # Find and mark the point of minimum drag
    if len(alphas_arr) > 0:
        if len(alphas_arr) >= 3 and k_cd != 0:
            alpha_min_cd = -j_cd / (2 * k_cd)
            alpha_min_cd = np.clip(alpha_min_cd, alphas_arr.min(), alphas_arr.max())
            cd_min_val = a_cd + j_cd * alpha_min_cd + k_cd * alpha_min_cd**2
        else:
            min_cd_idx = np.argmin(cd_arr)
            alpha_min_cd = alphas_arr[min_cd_idx]
            cd_min_val = cd_arr[min_cd_idx]
        # Vertical line at alpha of minimum drag
        axs[2].axvline(x=alpha_min_cd, color='green', linestyle='--', linewidth=1, alpha=0.7)
        # Horizontal line at minimum drag value
        axs[2].axhline(y=cd_min_val, color='green', linestyle='--', linewidth=1, alpha=0.7)
        # Add text label at top right of the vertical line with small offset from top
        y_lim = axs[2].get_ylim()
        y_range = y_lim[1] - y_lim[0]
        y_pos = y_lim[1] - 0.02 * y_range  # Small offset from top
        x_lim = axs[2].get_xlim()
        x_range = x_lim[1] - x_lim[0]
        x_offset = 0.01 * x_range
        axs[2].text(alpha_min_cd + x_offset, y_pos, f'α={alpha_min_cd:.2f}º\nmin $C_{{Di}}$ = {cd_min_val:.5f}', 
                   verticalalignment='top', horizontalalignment='left', color='green', fontsize=9)
        # Store for use in polar plot
        cl_at_min_cd = cl_intercept + cl_slope * alpha_min_cd if len(alphas_arr) > 1 else np.nan
    axs[2].text(0.02, 0.02, rf'$C_{{Di}} = {a_cd:.5f} + {j_cd:.5f}\,\alpha + {k_cd:.5f}\,\alpha^2$',
                transform=axs[2].transAxes,
                verticalalignment='bottom', horizontalalignment='left',
                color='black', fontsize=9)
    
    # Add vertical padding to make bottom formula readable
    y_lim = axs[2].get_ylim()
    y_range = y_lim[1] - y_lim[0]
    axs[2].set_ylim(y_lim[0] - 0.08 * y_range, y_lim[1])
    
    axs[2].set_title(r'$C_{Di}$')
    axs[2].set_xlabel('Angle of attack [º]')
    axs[2].grid(True, alpha=0.3)

    # x_cp: convert to % root chord and limit y-axis to +/- half-span in % root chord
    if hasattr(aero_wing, 'wing') and hasattr(aero_wing.wing, 'cr') and aero_wing.wing.cr != 0:
        xcp_plot = (xcp_arr / aero_wing.wing.cr) * 100
        xcp_ylabel = 'x_cp [% root chord]'
    else:
        xcp_plot = xcp_arr
        xcp_ylabel = 'x_cp'

    if int_mask.any():
        axs[3].plot(alphas_arr[int_mask], xcp_plot[int_mask], marker='o', linestyle='None', color='blue', markerfacecolor='none', markersize=4)
    # Mark plotted_alphas with red dots if provided
    if plotted_alphas is not None:
        plotted_mask = np.isin(alphas_arr, plotted_alphas)
        if plotted_mask.any():
            axs[3].plot(alphas_arr[plotted_mask], xcp_plot[plotted_mask], marker='o', linestyle='None', color='red', markerfacecolor='red', markersize=5)

    axs[3].set_ylim(-100, 200)

    # Find and mark the midpoint where xcp jumps (largest change)
    if len(alphas_arr) > 1:
        xcp_diffs = np.abs(np.diff(xcp_plot))
        max_jump_idx = np.argmax(xcp_diffs)
        # Midpoint between the two values where the jump occurs
        alpha_jump = (alphas_arr[max_jump_idx] + alphas_arr[max_jump_idx + 1]) / 2
        axs[3].axvline(x=alpha_jump, color='red', linestyle='--', linewidth=1, alpha=0.7)
        # Add text label at top right of the line with small offset from top
        y_lim = axs[3].get_ylim()
        y_range = y_lim[1] - y_lim[0]
        y_pos = y_lim[1] - 0.02 * y_range  # Small offset from top
        axs[3].text(alpha_jump, y_pos, f' α={alpha_jump:.2f}º', 
                   verticalalignment='top', horizontalalignment='left', color='red', fontsize=9)
    
    axs[3].set_title(r'$x_{cp}$')
    axs[3].set_xlabel('Angle of attack [º]')
    if xcp_ylabel == 'x_cp [% root chord]':
        axs[3].set_ylabel(r'$x_{cp}$ [% root chord]')
    else:
        axs[3].set_ylabel(r'$x_{cp}$')
    axs[3].grid(True, alpha=0.3)

    # cm0y and cma on same axes: markers only at integer alphas; interpolate cm0y
    if int_mask.any():
        l1 = axs[4].plot(alphas_arr[int_mask], cm0y_arr[int_mask], marker='o', linestyle='None', color='blue', markerfacecolor='none', markersize=4)[0]
        l2 = axs[4].plot(alphas_arr[int_mask], cma_arr[int_mask], marker='o', linestyle='None', markerfacecolor='none', markersize=4)[0]
    else:
        l1 = axs[4].plot([], [], marker='o', linestyle='None', markerfacecolor='none', markersize=4)[0]
        l2 = axs[4].plot([], [], marker='o', linestyle='None', markerfacecolor='none', markersize=4)[0]
    # Mark plotted_alphas with red dots if provided
    if plotted_alphas is not None:
        plotted_mask = np.isin(alphas_arr, plotted_alphas)
        if plotted_mask.any():
            axs[4].plot(alphas_arr[plotted_mask], cm0y_arr[plotted_mask], marker='o', linestyle='None', color='red', markerfacecolor='red', markersize=5)
            axs[4].plot(alphas_arr[plotted_mask], cma_arr[plotted_mask], marker='o', linestyle='None', color='red', markerfacecolor='red', markersize=5)
    if len(alphas_arr) > 1:
        cm_slope, cm_intercept = np.polyfit(alphas_arr, cm0y_arr, 1)
        cm_slope_rad = cm_slope * 180 / np.pi
        alpha_fit = np.linspace(alphas_arr.min(), alphas_arr.max(), 200)
        cm_fit = cm_intercept + cm_slope * alpha_fit
        axs[4].plot(alpha_fit, cm_fit, linestyle='--', linewidth=1.5, color='blue')
        axs[4].text(0.98, 0.98, rf'$ c_{{mα}}$ = {cm_slope_rad:.4f} 1/rad' + '\n' + rf'$c_{{m0}}$ = {cm_intercept:.4f}',
                    transform=axs[4].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    color=l1.get_color(), fontsize=9)
    # Mark the mean value of cma with a horizontal line
    if len(cma_arr) > 0:
        cma_mean = np.mean(cma_arr)
        cma_color = l2.get_color()
        axs[4].axhline(y=cma_mean, color=cma_color, linestyle='--', linewidth=1, alpha=0.7)
        # Add text label at left beneath the line with small offset
        x_lim = axs[4].get_xlim()
        y_lim = axs[4].get_ylim()
        y_range = y_lim[1] - y_lim[0]
        x_pos = x_lim[0]  # Left edge
        y_offset = 0.02 * y_range  # Small offset below the line
        axs[4].text(x_pos, cma_mean - y_offset, rf' $c_{{ma}}$={cma_mean:.4f}', 
                   verticalalignment='top', horizontalalignment='left', color=cma_color, fontsize=9)
    
    axs[4].set_title(r'$C_M$')
    axs[4].set_xlabel('Angle of attack [º]')
    axs[4].grid(True, alpha=0.3)

    # Polar plot: C_Di vs C_L with labels at multiples of 5º (integer alphas only)
    if int_mask.any():
        alpha_int_arr = np.round(alphas_arr[int_mask]).astype(int)
        named_mask = (alpha_int_arr % 5 == 0)

        # Unlabeled integer points (hollow)
        axs[5].plot(cd_arr[int_mask][~named_mask], cl_arr[int_mask][~named_mask], marker='o', linestyle='None',
                    color='blue', markerfacecolor='none', markersize=4)
        # Labeled points (solid)
        axs[5].plot(cd_arr[int_mask][named_mask], cl_arr[int_mask][named_mask], marker='o', linestyle='None',
                    color='blue', markerfacecolor='blue', markersize=4)

        cd_span = max(np.max(cd_arr[int_mask]) - np.min(cd_arr[int_mask]), 1e-8)
        x_offset = 0.025 * cd_span
        for cd_val, cl_val, alpha_int in zip(cd_arr[int_mask][named_mask], cl_arr[int_mask][named_mask], alpha_int_arr[named_mask]):
            # Check if this alpha is above the minimum CDi alpha
            if 'alpha_min_cd' in locals() and alpha_int > alpha_min_cd:
                # Place label to the left
                axs[5].text(cd_val - x_offset, cl_val, f'{alpha_int}º', fontsize=8, verticalalignment='bottom', horizontalalignment='right')
            else:
                # Place label to the right (default)
                axs[5].text(cd_val + x_offset, cl_val, f'{alpha_int}º', fontsize=8, verticalalignment='bottom', horizontalalignment='left')
    # Mark plotted_alphas with red dots if provided
    if plotted_alphas is not None:
        plotted_mask = np.isin(alphas_arr, plotted_alphas)
        if plotted_mask.any():
            axs[5].plot(cd_arr[plotted_mask], cl_arr[plotted_mask], marker='o', linestyle='None', color='red', markerfacecolor='red', markersize=5)

    # Quadratic interpolation: C_Di = a + j*C_L + k*C_L^2
    if len(cl_arr) >= 3:
        k_p, j_p, a_p = np.polyfit(cl_arr, cd_arr, 2)
        cl_fit_p = np.linspace(cl_arr.min(), cl_arr.max(), 300)
        cd_fit_p = a_p + j_p * cl_fit_p + k_p * cl_fit_p**2
        axs[5].plot(cd_fit_p, cl_fit_p, linestyle='--', linewidth=1.5, color='blue')
        axs[5].text(0.98, 0.98, rf'$C_{{Di}} = {a_p:.5f} + {j_p:.5f}\,C_L + {k_p:.5f}\,C_L^2$',
                    transform=axs[5].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    color='black', fontsize=9)
    # Add padding to y-axis for formula visibility
    y_lim_polar = axs[5].get_ylim()
    y_range_polar = y_lim_polar[1] - y_lim_polar[0]
    axs[5].set_ylim(y_lim_polar[0], y_lim_polar[1] + 0.08 * y_range_polar)
    # Add vertical line at CL corresponding to minimum CDi
    if len(alphas_arr) > 0 and 'cl_at_min_cd' in locals() and not np.isnan(cl_at_min_cd):
        axs[5].axhline(y=cl_at_min_cd, color='green', linestyle='--', linewidth=1, alpha=0.7)
        # Add vertical line at minimum CDi value
        if 'cd_min_val' in locals():
            axs[5].axvline(x=cd_min_val, color='green', linestyle='--', linewidth=1, alpha=0.7)
            # Add text label for CL value and min CDi at top right of vertical line
            y_lim = axs[5].get_ylim()
            y_pos = y_lim[1] - 0.02 * (y_lim[1] - y_lim[0])
            x_offset = 0.01 * (axs[5].get_xlim()[1] - axs[5].get_xlim()[0])
            axs[5].text(cd_min_val + x_offset, y_pos, f'$C_L$ = {cl_at_min_cd:.4f}\nmin $C_{{Di}}$ = {cd_min_val:.5f}', 
                       verticalalignment='top', horizontalalignment='left', color='green', fontsize=9)
    axs[5].set_title('Polar')
    axs[5].set_xlabel(r'$C_{Di}$')
    axs[5].set_ylabel(r'$C_L$')
    axs[5].grid(True, alpha=0.3)

    fig.tight_layout()

    # Prepare spanwise plots: use plotted_alphas if provided, otherwise choose a few alphas to overlay
    if plotted_alphas is not None:
        # Use the provided plotted_alphas for spanwise plots
        spanwise_alphas = [a for a in plotted_alphas if a in alphas]
    else:
        # Default behavior: prefer integer alpha values present in alpha_memory (e.g. -5, -4, ...).
        # This selects every integer angle that exists; if none are integer,
        # fall back to selecting up to 5 evenly spaced alphas.
        int_alphas = [a for a in alphas if float(a).is_integer()]
        if int_alphas:
            spanwise_alphas = int_alphas
        else:
            n = min(5, len(alphas))
            idx = np.linspace(0, len(alphas) - 1, n, dtype=int)
            spanwise_alphas = [alphas[i] for i in idx]

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

def plot_streamlines_3d(wing, alpha, stream_box_scale=(1, 1, 1), smoke_scale=(0.5, 0.5), 
                        stream_density=(5, 5), grid_resolution=(20, 20, 20), 
                        upstream_offset=0.5, max_length=3.0, step_size=0.05, show=True, tip_vortex=False):
    """
    Plot 3D streamlines around a wing showing induced flow patterns.
    
    Parameters:
    - wing: Aerdynamic_wing object with calculated alpha_memory
    - alpha: angle of attack to visualize
    - stream_box_scale: tuple (x, y, z) domain extent in each direction as fraction of wing span
    - smoke_scale: tuple (y, z) streamline injection area extent as fraction of wing span
    - stream_density: tuple (ny, nz) number of streamlines in y and z directions
    - grid_resolution: tuple (nx, ny, nz) grid points for velocity field calculation
    - upstream_offset: how far upstream (in span lengths) to start streamlines
    - max_length: maximum streamline length in span units
    - step_size: integration step size for streamlines
    - show: whether to display the plot
    - tip_vortex: if True, generate two smoke boxes at wing tips; if False, one centered box
    """
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import RegularGridInterpolator
    
    # Extract scale parameters
    x_scale, y_scale, z_scale = stream_box_scale
    smoke_y_scale, smoke_z_scale = smoke_scale
    
    # Get wing parameters
    b = wing.wing.b
    
    # Get circulation values for this alpha
    if alpha not in wing.alpha_memory:
        print(f"Alpha {alpha} not found in alpha_memory. Calculating...")
        wing.calculate(alpha)
    
    gamma = wing.alpha_memory[alpha]["C"]
    
    # Get vortex geometry from singularity model
    x1 = wing.model.x1
    y1 = wing.model.y1
    z1 = wing.model.z1
    x2 = wing.model.x2
    y2 = wing.model.y2
    z2 = wing.model.z2
    
    # Define computational domain
    x_grid = np.linspace(-upstream_offset * b, (x_scale - upstream_offset) * b, grid_resolution[0])
    y_grid = np.linspace(-y_scale * b / 2, y_scale * b / 2, grid_resolution[1])
    z_grid = np.linspace(-z_scale * b / 2, z_scale * b / 2, grid_resolution[2])
    
    # Calculate induced velocity field using Biot-Savart law
    print("Calculating velocity field...")
    u_field = np.zeros(grid_resolution)
    v_field = np.zeros(grid_resolution)
    w_field = np.zeros(grid_resolution)
    
    # Freestream velocity (horizontal for visualization)
    alpha_rad = alpha * np.pi / 180
    u_inf = 1.0
    w_inf = 0.0
    
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            for k, z in enumerate(z_grid):
                vel = np.zeros(3)
                
                # Contribution from each horseshoe vortex
                for idx in range(len(gamma)):
                    g = gamma[idx]
                    
                    # Left trailing vortex (from far upstream to point 1)
                    xA = np.array([x1[idx] + 1e5, y1[idx], 0.0])
                    xB = np.array([x1[idx], y1[idx], 0.0])
                    ABv = xB - xA
                    AB = np.linalg.norm(ABv)
                    BCv = np.array([x - xB[0], y - xB[1], z - xB[2]])
                    BC = np.linalg.norm(BCv)
                    d = np.dot(ABv, BCv) / AB
                    r = np.sqrt(BC**2 - d**2 + 1e-8)
                    cross = np.cross(ABv, BCv) + 1e-8
                    cross = cross / np.linalg.norm(cross)
                    v_left = cross * g / (4 * np.pi * r) * ((d + AB) / np.sqrt(r**2 + (d + AB)**2) - d / np.sqrt(r**2 + d**2))
                    vel += v_left
                    
                    # Bound vortex (from point 1 to point 2)
                    xA = np.array([x1[idx], y1[idx], 0.0])
                    xB = np.array([x2[idx], y2[idx], 0.0])
                    ABv = xB - xA
                    AB = np.linalg.norm(ABv)
                    BCv = np.array([x - xB[0], y - xB[1], z - xB[2]])
                    BC = np.linalg.norm(BCv)
                    d = np.dot(ABv, BCv) / AB
                    r = np.sqrt(BC**2 - d**2 + 1e-8)
                    cross = np.cross(ABv, BCv) + 1e-8
                    cross = cross / np.linalg.norm(cross)
                    v_bound = cross * g / (4 * np.pi * r) * ((d + AB) / np.sqrt(r**2 + (d + AB)**2) - d / np.sqrt(r**2 + d**2))
                    vel += v_bound
                    
                    # Right trailing vortex (from far downstream to point 2) - opposite circulation
                    xA = np.array([x2[idx] + 1e5, y2[idx], 0.0])
                    xB = np.array([x2[idx], y2[idx], 0.0])
                    ABv = xB - xA
                    AB = np.linalg.norm(ABv)
                    BCv = np.array([x - xB[0], y - xB[1], z - xB[2]])
                    BC = np.linalg.norm(BCv)
                    d = np.dot(ABv, BCv) / AB
                    r = np.sqrt(BC**2 - d**2 + 1e-8)
                    cross = np.cross(ABv, BCv) + 1e-8
                    cross = cross / np.linalg.norm(cross)
                    v_right = -cross * g / (4 * np.pi * r) * ((d + AB) / np.sqrt(r**2 + (d + AB)**2) - d / np.sqrt(r**2 + d**2))
                    vel += v_right
                
                u_field[i, j, k] = vel[0] + u_inf
                v_field[i, j, k] = vel[1]
                w_field[i, j, k] = vel[2] + w_inf
    
    # Plot velocity magnitude field (optional)
    if show:
        print("Plotting velocity field...")
        speed_field = np.sqrt(u_field**2 + v_field**2 + w_field**2)
        
        fig_vel = plt.figure(figsize=(14, 10))
        ax_vel = fig_vel.add_subplot(111, projection='3d')
        
        # Remove background, grid, and axis
        ax_vel.axis('off')
        ax_vel.grid(False)
        ax_vel.xaxis.pane.fill = False
        ax_vel.yaxis.pane.fill = False
        ax_vel.zaxis.pane.fill = False
        ax_vel.xaxis.pane.set_edgecolor('none')
        ax_vel.yaxis.pane.set_edgecolor('none')
        ax_vel.zaxis.pane.set_edgecolor('none')
        
        # Create meshgrid for plotting
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        
        # Flatten arrays for scatter plot
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        speed_flat = speed_field.flatten()
        
        # Plot with color representing speed
        scatter = ax_vel.scatter(x_flat, y_flat, z_flat, c=speed_flat, cmap='jet', 
                                s=10, alpha=0.3, vmin=speed_flat.min(), vmax=speed_flat.max())
        
        # Add colorbar
        cbar = fig_vel.colorbar(scatter, ax=ax_vel, pad=0.1, shrink=0.8)
        cbar.set_label('Speed magnitude', rotation=270, labelpad=20)
        
        # Plot wing surface (rotated by alpha for visualization)
        if hasattr(wing.wing, 'x_mesh'):
            x_mesh = wing.wing.x_mesh
            y_mesh = wing.wing.y_mesh
            z_mesh = wing.wing.z_mesh
            
            # Rotate wing by alpha angle for visualization
            x_rot = x_mesh * np.cos(alpha_rad) + z_mesh * np.sin(alpha_rad)
            z_rot = -x_mesh * np.sin(alpha_rad) + z_mesh * np.cos(alpha_rad)
            
            # Plot wing as surface
            ax_vel.plot_surface(x_rot, y_mesh, z_rot, color='gray', alpha=0.6, edgecolor='black', linewidth=0.5)
        
        ax_vel.set_title(f'Velocity Field Magnitude (α = {alpha}°)')
        
        # Set axis limits with equal aspect ratio
        x_min, x_max = -upstream_offset * b, (x_scale - upstream_offset) * b
        y_min, y_max = -y_scale * b / 2, y_scale * b / 2
        z_min, z_max = -z_scale * b / 2, z_scale * b / 2
        
        max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
        mid_z = (z_max + z_min) * 0.5
        
        ax_vel.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_vel.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_vel.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.draw()
        plt.pause(0.001)  # Allow plot to display without blocking
    
    # Create interpolator for velocity field
    print("Creating velocity interpolator...")
    u_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), u_field, bounds_error=False, fill_value=u_inf)
    v_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), v_field, bounds_error=False, fill_value=0)
    w_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), w_field, bounds_error=False, fill_value=w_inf)
    
    # Generate starting points for streamlines (upstream plane)
    print("Integrating streamlines...")
    x_start = -upstream_offset * b
    z_starts = np.linspace(-smoke_z_scale * b / 2, smoke_z_scale * b / 2, stream_density[1])
    
    # Define y positions based on tip_vortex mode
    if tip_vortex:
        # Two smoke boxes centered at wing tips
        y_left_center = -b / 2
        y_right_center = b / 2
        y_starts_left = np.linspace(y_left_center - smoke_y_scale * b / 2, 
                                     y_left_center + smoke_y_scale * b / 2, 
                                     stream_density[0])
        y_starts_right = np.linspace(y_right_center - smoke_y_scale * b / 2, 
                                      y_right_center + smoke_y_scale * b / 2, 
                                      stream_density[0])
        y_starts_list = [y_starts_left, y_starts_right]
    else:
        # Single centered smoke box
        y_starts = np.linspace(-smoke_y_scale * b / 2, smoke_y_scale * b / 2, stream_density[0])
        y_starts_list = [y_starts]
    
    streamlines = []
    for y_starts in y_starts_list:
        for y_s in y_starts:
            for z_s in z_starts:
                # Integrate streamline using RK4
                streamline = [[x_start, y_s, z_s]]
                
                for _ in range(int(max_length * b / step_size)):
                    pos = streamline[-1]
                    
                    # Check if outside domain
                    if (pos[0] < x_grid[0] or pos[0] > x_grid[-1] or
                        pos[1] < y_grid[0] or pos[1] > y_grid[-1] or
                        pos[2] < z_grid[0] or pos[2] > z_grid[-1]):
                        break
                    
                    # RK4 integration
                    k1 = np.array([u_interp(pos)[0], v_interp(pos)[0], w_interp(pos)[0]])
                    k2 = np.array([u_interp(pos + 0.5 * step_size * k1)[0], 
                                  v_interp(pos + 0.5 * step_size * k1)[0],
                                  w_interp(pos + 0.5 * step_size * k1)[0]])
                    k3 = np.array([u_interp(pos + 0.5 * step_size * k2)[0],
                                  v_interp(pos + 0.5 * step_size * k2)[0],
                                  w_interp(pos + 0.5 * step_size * k2)[0]])
                    k4 = np.array([u_interp(pos + step_size * k3)[0],
                                  v_interp(pos + step_size * k3)[0],
                                  w_interp(pos + step_size * k3)[0]])
                    
                    vel = (k1 + 2*k2 + 2*k3 + k4) / 6
                    vel_mag = np.linalg.norm(vel)
                    
                    if vel_mag < 1e-8:
                        break
                    
                    new_pos = pos + step_size * vel / vel_mag * vel_mag
                    streamline.append(new_pos)
                
                if len(streamline) > 2:
                    streamlines.append(np.array(streamline))
    
    # Plot the results
    print("Plotting streamlines...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove background, grid, and axis
    ax.axis('off')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Plot streamlines
    for streamline in streamlines:
        ax.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2], 
               'b-', alpha=0.6, linewidth=0.8)
    
    # Plot wing surface (rotated by alpha for visualization)
    if hasattr(wing.wing, 'x_mesh'):
        x_mesh = wing.wing.x_mesh
        y_mesh = wing.wing.y_mesh
        z_mesh = wing.wing.z_mesh
        
        # Rotate wing by alpha angle for visualization
        x_rot = x_mesh * np.cos(alpha_rad) + z_mesh * np.sin(alpha_rad)
        z_rot = -x_mesh * np.sin(alpha_rad) + z_mesh * np.cos(alpha_rad)
        
        # Plot wing as surface
        ax.plot_surface(x_rot, y_mesh, z_rot, color='gray', alpha=0.6, edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'3D Streamlines around wing (α = {alpha}°)')
    
    # Set axis limits with equal aspect ratio
    x_min, x_max = -upstream_offset * b, (x_scale - upstream_offset) * b
    y_min, y_max = -y_scale * b / 2, y_scale * b / 2
    z_min, z_max = -z_scale * b / 2, z_scale * b / 2
    
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()
    
    return fig, ax, streamlines