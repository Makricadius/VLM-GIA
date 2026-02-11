from geometry import Wing
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def generate_wing_nodes(span, root_le, tip_le, sweep_deg, dihedral_deg=2.0, chord_root=1.5, chord_tip=0.6, torsion_root_deg=0.0, torsion_tip_deg=-6.0, n_sections=10):
	"""
	Generate leading- and trailing-edge nodes along the wing span with sweep and dihedral.

	span: total span (same units as coords)
	root_le: (x,y,z) tuple for root leading edge
	tip_le: (x,y,z) tuple for tip leading edge
	sweep_deg: sweep angle in degrees (positive sweeps towards +x)
	dihedral_deg: dihedral angle in degrees (positive raises tip)
	chord_root/chord_tip: chord lengths at root and tip (positive toward +x)
	n_sections: number of spanwise segments (nodes = n_sections+1)
	Returns (leading_nodes, trailing_nodes) lists of (x,y,z) from root to tip.
	"""
	sx, sy, sz = root_le
	tx, ty, tz = tip_le
	sweep_rad = math.radians(sweep_deg)
	dihedral_rad = math.radians(dihedral_deg)
	leading = []
	trailing = []
	for i in range(n_sections + 1):
		t = i / n_sections
		# base interpolation between provided root/tip coords (signed)
		x_base = sx + t * (tx - sx)
		y = sy + t * (ty - sy)
		z_base = sz + t * (tz - sz)
		# use absolute spanwise distance for sweep/dihedral magnitude
		local_y_abs = abs(y - sy)
		# sweep and dihedral offsets (same sign for both halves)
		sweep_offset = math.tan(sweep_rad) * local_y_abs
		dihedral_offset = math.tan(dihedral_rad) * local_y_abs
		x_le = x_base + sweep_offset
		z_le = z_base + dihedral_offset
		# chord interpolation
		chord = chord_root + t * (chord_tip - chord_root)
		# torsion interpolation (degrees -> radians)
		torsion_deg = torsion_root_deg + t * (torsion_tip_deg - torsion_root_deg)
		torsion_rad = math.radians(torsion_deg)
		# apply torsion: rotate chord vector about local spanwise axis (y)
		# chord vector initially along +x; rotated components along x and z
		x_te = x_le + chord * math.cos(torsion_rad)
		z_te = z_le + chord * math.sin(torsion_rad)
		leading.append((x_le, y, z_le))
		trailing.append((x_te, y, z_te))
	return leading, trailing


def generate_symmetric_wing(span, root_le_center, sweep_deg, dihedral_deg=2.0, chord_root=1.5, chord_tip=0.6, torsion_root_deg=0.0, torsion_tip_deg=-6.0, n_sections=10):
	"""
	Generate symmetric left and right wing nodes about the center (root at centerline).
	root_le_center: (x,y,z) for the wing root at the centerline (usually y=0)
	Returns dict with keys `right` and `left`, each a tuple (leading, trailing).
	"""
	sx, sy, sz = root_le_center
	# right tip (positive y) and left tip (negative y)
	right_tip = (sx, sy + span, sz)
	left_tip = (sx, sy - span, sz)
	r_le, r_te = generate_wing_nodes(span, root_le_center, right_tip, sweep_deg, dihedral_deg, chord_root, chord_tip, torsion_root_deg, torsion_tip_deg, n_sections)
	l_le, l_te = generate_wing_nodes(span, root_le_center, left_tip, sweep_deg, dihedral_deg, chord_root, chord_tip, torsion_root_deg, torsion_tip_deg, n_sections)
	return {"right": (r_le, r_te), "left": (l_le, l_te)}


if __name__ == "__main__":
	my_wing = Wing("Trapezoidal")
	print(my_wing.type)

	# Example usage: generate wing nodes (LE + TE) with small dihedral
	# base geometry
	span = 10.0
	# increase span by 20%
	span *= 1.2
	root_le = (0.0, 0.0, 0.0)
	tip_le = (0.0, span, 0.0)
	sweep = 15.0  # degrees
	# decrease sweep by 2 degrees
	sweep -= 2.0
	# dihedral set to 8 degrees
	dihedral = 8.0
	# double root chord and set tip chord
	chord_root = 1.5 * 2.0
	chord_tip = 0.6
	# torsion: linear from root 0 deg to tip -6 deg
	torsion_root = 0.0
	torsion_tip = 10.0
	leading, trailing = generate_wing_nodes(span, root_le, tip_le, sweep, dihedral_deg=dihedral, chord_root=chord_root, chord_tip=chord_tip, torsion_root_deg=torsion_root, torsion_tip_deg=torsion_tip, n_sections=10)

	print("Leading-edge nodes:")
	for n in leading:
		print(n)
	print("Trailing-edge nodes:")
	for n in trailing:
		print(n)

	# 3D plot of wing
	try:
		# build symmetric wing
		sym = generate_symmetric_wing(span, root_le, sweep, dihedral, chord_root, chord_tip, torsion_root, torsion_tip, n_sections=10)
		r_le, r_te = sym['right']
		l_le, l_te = sym['left']
		# collect coordinates per side (do NOT join tips between sides)
		r_xs_le = [n[0] for n in r_le]
		r_ys = [n[1] for n in r_le]
		r_zs_le = [n[2] for n in r_le]
		r_xs_te = [n[0] for n in r_te]
		r_zs_te = [n[2] for n in r_te]
		l_xs_le = [n[0] for n in l_le]
		l_ys = [n[1] for n in l_le]
		l_zs_le = [n[2] for n in l_le]
		l_xs_te = [n[0] for n in l_te]
		l_zs_te = [n[2] for n in l_te]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		# plot leading and trailing edges for each side separately
		ax.plot(r_xs_le, r_ys, r_zs_le, marker='o', linestyle='-', label='LE (right)')
		ax.plot(r_xs_te, r_ys, r_zs_te, marker='o', linestyle='--', label='TE (right)')
		ax.plot(l_xs_le, l_ys, l_zs_le, marker='o', linestyle='-', label='LE (left)')
		ax.plot(l_xs_te, l_ys, l_zs_te, marker='o', linestyle='--', label='TE (left)')
		# quarter-chord (25%) along chord for visualization
		r_qc_xs = [r_le[i][0] + 0.25*(r_te[i][0]-r_le[i][0]) for i in range(len(r_le))]
		r_qc_zs = [r_le[i][2] + 0.25*(r_te[i][2]-r_le[i][2]) for i in range(len(r_le))]
		l_qc_xs = [l_le[i][0] + 0.25*(l_te[i][0]-l_le[i][0]) for i in range(len(l_le))]
		l_qc_zs = [l_le[i][2] + 0.25*(l_te[i][2]-l_le[i][2]) for i in range(len(l_le))]
		# plot quarter-chord lines
		ax.plot(r_qc_xs, r_ys, r_qc_zs, marker='x', linestyle='-', color='magenta', label='25% (right)')
		ax.plot(l_qc_xs, l_ys, l_qc_zs, marker='x', linestyle='-', color='cyan', label='25% (left)')

		# --- Tail generation ---
		# tail: no dihedral, tip torsion -1 deg, span 20% of wing half-span, chord scaled to 30%
		tail_span = span * 0.2
		tail_chord_root = chord_root * 0.3
		tail_chord_tip = chord_tip * 0.3
		tail_dihedral = 0.0
		tail_torsion_root = 0.0
		tail_torsion_tip = -1.0
		# place tail root behind the wing by 0.7*span plus extra 0.2*span gap (total 0.9*span)
		wing_trailing_root_x = trailing[0][0]
		tail_root_x = wing_trailing_root_x + 0.9 * span
		tail_root = (tail_root_x, 0.0, 0.0)
		# generate symmetric tail about centerline with 20deg sweep, fewer partitions
		tail_sym = generate_symmetric_wing(tail_span, tail_root, sweep_deg=20.0, dihedral_deg=tail_dihedral, chord_root=tail_chord_root, chord_tip=tail_chord_tip, torsion_root_deg=tail_torsion_root, torsion_tip_deg=tail_torsion_tip, n_sections=5)
		t_le, t_te = tail_sym['right']
		T_le, T_te = tail_sym['left']
		# collect tail coords for plotting
		t_r_xs_le = [n[0] for n in t_le]
		t_r_ys = [n[1] for n in t_le]
		t_r_zs_le = [n[2] for n in t_le]
		t_r_xs_te = [n[0] for n in t_te]
		t_r_zs_te = [n[2] for n in t_te]
		t_l_xs_le = [n[0] for n in T_le]
		t_l_ys = [n[1] for n in T_le]
		t_l_zs_le = [n[2] for n in T_le]
		t_l_xs_te = [n[0] for n in T_te]
		t_l_zs_te = [n[2] for n in T_te]
		# plot tail edges
		ax.plot(t_r_xs_le, t_r_ys, t_r_zs_le, marker='o', linestyle='-', color='brown', label='Tail LE (right)')
		ax.plot(t_r_xs_te, t_r_ys, t_r_zs_te, marker='o', linestyle='--', color='saddlebrown', label='Tail TE (right)')
		ax.plot(t_l_xs_le, t_l_ys, t_l_zs_le, marker='o', linestyle='-', color='brown', label='Tail LE (left)')
		ax.plot(t_l_xs_te, t_l_ys, t_l_zs_te, marker='o', linestyle='--', color='saddlebrown', label='Tail TE (left)')
		# quarter-chord for tail
		t_r_qc_xs = [t_le[i][0] + 0.25*(t_te[i][0]-t_le[i][0]) for i in range(len(t_le))]
		t_r_qc_zs = [t_le[i][2] + 0.25*(t_te[i][2]-t_le[i][2]) for i in range(len(t_le))]
		t_l_qc_xs = [T_le[i][0] + 0.25*(T_te[i][0]-T_le[i][0]) for i in range(len(T_le))]
		t_l_qc_zs = [T_le[i][2] + 0.25*(T_te[i][2]-T_le[i][2]) for i in range(len(T_le))]
		ax.plot(t_r_qc_xs, t_r_ys, t_r_qc_zs, marker='x', linestyle='-', color='orange', label='Tail 25% (right)')
		ax.plot(t_l_qc_xs, t_l_ys, t_l_qc_zs, marker='x', linestyle='-', color='yellow', label='Tail 25% (left)')
		# connect LE-TE across chord to show surface for each side separately
		for i in range(len(r_le)):
			# right side chord lines
			rx_lx, rx_ly, rx_lz = r_le[i]
			rx_tx, rx_ty, rx_tz = r_te[i]
			ax.plot([rx_lx, rx_tx], [rx_ly, rx_ty], [rx_lz, rx_tz], color='gray', linestyle='-')
			# left side chord lines
			lx_lx, lx_ly, lx_lz = l_le[i]
			lx_tx, lx_ty, lx_tz = l_te[i]
			ax.plot([lx_lx, lx_tx], [lx_ly, lx_ty], [lx_lz, lx_tz], color='gray', linestyle='-')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_title('Symmetric Wing (LE & TE) with Dihedral')
		ax.legend()
		# make axes proportional
		try:
			x_vals = r_xs_le + r_xs_te + l_xs_le + l_xs_te + r_qc_xs + l_qc_xs + t_r_xs_le + t_r_xs_te + t_l_xs_le + t_l_xs_te + t_r_qc_xs + t_l_qc_xs
			y_vals = r_ys + l_ys + t_r_ys + t_l_ys
			z_vals = r_zs_le + r_zs_te + l_zs_le + l_zs_te + r_qc_zs + l_qc_zs + t_r_zs_le + t_r_zs_te + t_l_zs_le + t_l_zs_te + t_r_qc_zs + t_l_qc_zs
			x_range = max(x_vals) - min(x_vals)
			y_range = max(y_vals) - min(y_vals)
			z_range = max(z_vals) - min(z_vals)
			# avoid zero-range
			if x_range == 0: x_range = 1.0
			if y_range == 0: y_range = 1.0
			if z_range == 0: z_range = 1.0
			ax.set_box_aspect((x_range, y_range, z_range))
		except Exception:
			pass
		plt.show()
	except Exception as e:
		print('Plotting failed:', e)